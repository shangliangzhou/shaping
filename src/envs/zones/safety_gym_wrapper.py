from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys
import re
import types

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, WrapperObsType
from gymnasium.spaces import Box



# 允许直接运行本文件自测
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[2]  # .../src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    print("[DEBUG] add SRC_DIR to sys.path:", SRC_DIR)
# Deep-LTL 的命题赋值工具
from ltl.logic import Assignment
# ---- MuJoCo 官方包（2.x） ----
try:
    import mujoco as mj
except Exception as e:
    mj = None
    print("[WARN] mujoco import failed:", e)

COLOR_ORDER = ['blue', 'green', 'yellow', 'magenta']
# ================= 数据结构 =================


@dataclass
class ZoneGeom:
    """一个 zone 几何的底层真值（世界系）。"""
    name: str
    gid: Optional[int]             # 通过 layout 构造时可能没有 gid
    color: str                     # 'blue'|'green'|'yellow'|'magenta'|...
    kind: str                      # 'circle'|'box'|'other'
    center_xy: np.ndarray          # (x, y)
    radius: Optional[float] = None             # circle 用
    half_extents: Optional[np.ndarray] = None  # (hx, hy) for box


# ================= 小工具 =================

def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=float)

def _safe_float(x) -> Optional[float]:
    try:
        a = np.array(x)
        return float(a.reshape(-1)[0])
    except Exception:
        return None

def _colors_default_from_obs_space(obs_space: spaces.Dict) -> List[str]:
    """从观测空间键名推断颜色集合（若推断不到则给默认四色）。"""
    colors = set()
    for k in getattr(obs_space, "spaces", {}).keys():
        m = re.match(r"(blue|green|yellow|magenta|red)", k)
        if m and k.endswith("zones_lidar"):
            colors.add(m.group(1))
    if not colors:
        colors.update(["blue", "green", "yellow", "magenta"])
    return sorted(colors)

def _mjt_val(name: str, default: int) -> int:
    try:
        return int(getattr(mj.mjtGeom, name))
    except Exception:
        return int(default)

def _type_name(x: Any) -> str:
    try:
        return type(x).__name__
    except Exception:
        return str(type(x))


# ================= 主包装器 =================

class SafetyGymWrapper(gymnasium.Wrapper):
    """
    将 Safety-Gymnasium LTL 场景封装到统一 API：
      - 直接从 MuJoCo model/data（或 layout 兜底）抽取 zone 几何真值（中心与尺寸）。
      - 输出几何命题 propositions_geom 与带符号边界距 sdist_{color}。
      - 保留原 env 的 info['propositions']（基于 cost_zones_*），便于对照。
    """

    # ----------- 初始化 -----------
    def __init__(self, env: Any, wall_sensor: bool = True):
        super().__init__(env)
        # 渲染参数
        self.render_parameters.camera_name = 'track'
        self.render_parameters.width = 256
        self.render_parameters.height = 256

        # 颜色集合
        # if isinstance(env.observation_space, spaces.Dict):
        #     self.colors: List[str] = _colors_default_from_obs_space(env.observation_space)
        # else:
        #     raise TypeError("SafetyGymWrapper expects Dict observation space.")
        # 旧代码可能是从 observation_space 键名推断颜色
        # 替换为以下固定顺序逻辑
        if isinstance(env.observation_space, spaces.Dict):
            # 观测里有哪些颜色键（如 blue_zones_lidar 等）
            obs_keys = list(env.observation_space.spaces.keys())
            discovered = []
            for c in COLOR_ORDER:
                # 观察里存在这个颜色相关键，就纳入；否则跳过
                if any(k.startswith(c) for k in obs_keys):
                    discovered.append(c)
            # 如果一个都发现不了，就用默认完整顺序（确保一致）
            self.colors = discovered if discovered else COLOR_ORDER[:]
        else:
            raise TypeError("SafetyGymWrapper expects Dict observation space.")


        # 扩展观测空间（添加墙传感器占位）
        self.observation_space = spaces.Dict(env.observation_space)  # 拷贝
        if wall_sensor:
            self.observation_space['wall_sensor'] = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float64)

        # 句柄缓存
        self._model = None
        self._data = None
        self._agent_body_id: Optional[int] = None
        self._zones_truth: Dict[str, List[ZoneGeom]] = {c: [] for c in self.colors}

        # 初始化时抽取一次几何（很多任务 reset 后位置会变化，reset 里还会再抽）
        self._introspect_geoms_from_backend()

    # ----------- 顶层：从后端抽取 -----------
    def _introspect_geoms_from_backend(self) -> None:
        """先找 (model,data)，成功则走 MuJoCo 抽取；否则尝试 layout 兜底。"""
        ok = self._locate_model_data(verbose=True)
        if ok:
            self._extract_from_mujoco()
            # 若一个都没抽到，再尝试 layout 兜底（部分任务名称规则不同）
            if sum(len(v) for v in self._zones_truth.values()) == 0:
                self._extract_from_layout()
        else:
            # 完全找不到 mj 句柄，直接尝试 layout 兜底
            self._extract_from_layout()

    # ----------- 定位 (model, data) -----------
    def _locate_model_data(self, verbose: bool = False) -> bool:
        """
        在 env / env.unwrapped / .task / .engine / .sim 等层级里找 (model,data)。
        找不到则遍历 __dict__，按类型名包含 'MjModel' / 'MjData' 自动抓取。
        """
        self._model, self._data = None, None
        if mj is None:
            if verbose:
                print("[WARN] mujoco not available in current Python env.")
            return False

        # 常见候选路径（逐一尝试）
        u = getattr(self, "unwrapped", self.env)
        candidates: List[Tuple[Any, str]] = [
            (u, "model"), (u, "data"),
            (u, "sim"),   # 可能是 MjSim 或含 model/data 的对象
            (u, "task"), (u, "engine"), (u, "robot"),
        ]
        objs = [u]
        for base, attr in list(candidates):
            try:
                x = getattr(base, attr)
                if x is not None and x not in objs:
                    objs.append(x)
            except Exception:
                pass

        # 在一组对象里尝试直接拿 .model / .data
        for obj in objs:
            try:
                if hasattr(obj, "model") and hasattr(obj, "data"):
                    self._model = getattr(obj, "model")
                    self._data = getattr(obj, "data")
            except Exception:
                pass
            if self._model is not None and self._data is not None:
                break

        # 若还没取到，展开一层对象的 __dict__ 做浅遍历
        def _shallow_expand(obj: Any) -> List[Any]:
            out = []
            try:
                for k, v in vars(obj).items():
                    if isinstance(v, (types.ModuleType, types.FunctionType)):
                        continue
                    out.append(v)
            except Exception:
                pass
            return out

        if self._model is None or self._data is None:
            pool = list(objs)
            for obj in list(objs):
                pool.extend(_shallow_expand(obj))
            # 直接根据类型名匹配
            for x in pool:
                tn = _type_name(x)
                if self._model is None and ("MjModel" in tn or "mjModel" in tn or "Model" == tn):
                    self._model = x
                if self._data is None and ("MjData" in tn or "mjData" in tn or "Data" == tn):
                    self._data = x
                if self._model is not None and self._data is not None:
                    break

        ok = (self._model is not None and self._data is not None)

        if verbose:
            print("[DEBUG] locate (model,data) ->", ok)
            if not ok:
                # 打印可见属性，帮助定位
                def _dir_brief(o: Any) -> List[str]:
                    try:
                        return sorted([k for k in dir(o) if not k.startswith("__")])
                    except Exception:
                        return []
                print("  - unwrapped type:", _type_name(u))
                print("  - unwrapped attrs:", _dir_brief(u)[:50], "...")
            else:
                print("  - model type:", _type_name(self._model))
                print("  - data  type:", _type_name(self._data))

        # agent body id（尽量鲁棒）
        self._agent_body_id = None
        if ok:
            m = self._model
            # 常见 agent 名称
            for nm in ("agent", "robot", "robot0", "torso", "point", "car"):
                try:
                    bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, nm)
                    if int(bid) >= 0:
                        self._agent_body_id = int(bid)
                        break
                except Exception:
                    pass
            # 兜底：选第一个拥有自由关节的 body
            if self._agent_body_id is None:
                try:
                    for bid in range(m.nbody):
                        bname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, bid) or ""
                        if any(t in (bname or "").lower() for t in ("agent", "robot", "point", "car")):
                            self._agent_body_id = bid
                            break
                except Exception:
                    pass

        return ok

    # ----------- 从 MuJoCo 模型抽取几何真值 -----------
    def _extract_from_mujoco(self) -> None:
        self._zones_truth = {c: [] for c in self.colors}
        m, d = self._model, self._data
        if m is None or d is None:
            print("[WARN] _extract_from_mujoco called without model/data.")
            return

        T_SPHERE   = _mjt_val("mjGEOM_SPHERE",   2)
        T_CYLINDER = _mjt_val("mjGEOM_CYLINDER", 5)
        T_CAPSULE  = _mjt_val("mjGEOM_CAPSULE",  3)
        T_BOX      = _mjt_val("mjGEOM_BOX",      6)

        pat_tag = re.compile(r"(zone|zones|ltl)", re.I)
        pat_color = re.compile(r"(blue|green|yellow|magenta|red)", re.I)

        ngeom = int(getattr(m, "ngeom", 0))
        found = 0
        for gid in range(ngeom):
            try:
                gname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, gid) or ""
            except Exception:
                continue
            if not gname:
                continue
            name_l = gname.lower()
            if not pat_tag.search(name_l):
                continue
            m_color = pat_color.search(name_l)
            if not m_color:
                continue
            color = m_color.group(1).lower()
            if color not in self._zones_truth:
                self._zones_truth[color] = []

            try:
                cxy = _as_np(d.geom_xpos[gid][:2])
            except Exception:
                continue

            gtype = int(m.geom_type[gid])
            gsize = _as_np(m.geom_size[gid])

            if gtype in (T_SPHERE, T_CYLINDER, T_CAPSULE):
                r = float(gsize[0]) if gsize.size > 0 else None
                zg = ZoneGeom(gname, gid, color, "circle", cxy, radius=r)
            elif gtype == T_BOX:
                hx = float(gsize[0]) if gsize.size > 0 else None
                hy = float(gsize[1]) if gsize.size > 1 else None
                if hx is None or hy is None:
                    continue
                zg = ZoneGeom(gname, gid, color, "box", cxy, half_extents=np.array([hx, hy], dtype=float))
            else:
                zg = ZoneGeom(gname, gid, color, "other", cxy)

            self._zones_truth[color].append(zg)
            found += 1

        print("[INFO] (mujoco) zones truth loaded:",
              {k: len(v) for k, v in self._zones_truth.items()}, "(total:", found, ")")

    # ----------- 从 layout 兜底抽取（若实现） -----------
    def _extract_from_layout(self) -> None:
        self._zones_truth = {c: [] for c in self.colors}
        u = getattr(self, "unwrapped", self.env)
        layout = None
        try:
            layout = getattr(getattr(u, "world_info", None), "layout", None)
        except Exception:
            layout = None

        if not layout:
            print("[WARN] no (model,data) and no world_info.layout; cannot extract geometric zones.")
            return

        # 期望 layout 里有 zones / geoms / objects 等字段；做一些通用解析
        colors_seen = 0
        def _add(color: str, kind: str, center: Tuple[float, float],
                 radius: Optional[float] = None, half: Optional[Tuple[float, float]] = None,
                 name: str = "layout_zone"):
            nonlocal colors_seen
            if color not in self._zones_truth:
                self._zones_truth[color] = []
            self._zones_truth[color].append(
                ZoneGeom(name=name, gid=None, color=color, kind=kind,
                         center_xy=_as_np(center), radius=radius,
                         half_extents=(None if half is None else _as_np(half)))
            )
            colors_seen += 1

        # 粗略尝试一些常见键
        try:
            # 例如：layout["zones"] = [{"color":"blue","shape":"circle","pos":[x,y],"size":[r]}]
            if isinstance(layout, dict) and "zones" in layout:
                for z in layout["zones"]:
                    color = str(z.get("color", "")).lower()
                    shape = str(z.get("shape", "")) or str(z.get("kind", ""))
                    pos = z.get("pos") or z.get("center") or [0.0, 0.0]
                    size = z.get("size") or z.get("half_extents") or []
                    if isinstance(size, (list, tuple)) and len(size) == 1:
                        _add(color, "circle", (pos[0], pos[1]), radius=float(size[0]))
                    elif isinstance(size, (list, tuple)) and len(size) >= 2:
                        _add(color, "box", (pos[0], pos[1]), half=(float(size[0]), float(size[1])))
        except Exception:
            pass

        print("[INFO] (layout) zones truth loaded:",
              {k: len(v) for k, v in self._zones_truth.items()}, "(added:", colors_seen, ")")

    # ----------- 几何 → 距离/命题 -----------
    def _get_agent_xy(self) -> Optional[np.ndarray]:
        """读取 agent 世界坐标 (x, y)。"""
        # if self._data is None or self._model is None or self._agent_body_id is None:
        #     return None
        # try:
        #     return _as_np(self._data.xpos[self._agent_body_id][:2])
        # except Exception:
        #     return None
        #增加一个兜底（如果 MuJoCo 取不到，就读 unwrapped.agent_pos）
        if self._data is not None and self._model is not None and self._agent_body_id is not None:
            try:
                    return np.asarray(self._data.xpos[self._agent_body_id][:2], dtype=float)
            except Exception:
                    pass
        # 兜底：部分 builder 会把平面位置放在 agent_pos = [x,y,z]
        try:
            ap = getattr(self.unwrapped, 'agent_pos', None)
            if ap is not None and len(ap) >= 2:
                return np.asarray(ap[:2], dtype=float)
        except Exception:
            pass
        return None

    @staticmethod
    def _sdist_circle(ax: float, ay: float, cx: float, cy: float, r: float) -> float:
        dist = np.hypot(ax - cx, ay - cy)
        return r - dist  # inside: 正；outside: 负

    @staticmethod
    def _sdist_box(ax: float, ay: float, cx: float, cy: float, hx: float, hy: float) -> float:
        dx, dy = ax - cx, ay - cy
        ox, oy = abs(dx) - hx, abs(dy) - hy
        if ox <= 0.0 and oy <= 0.0:
            return float(min(hx - abs(dx), hy - abs(dy)))
        else:
            return -float(np.hypot(max(ox, 0.0), max(oy, 0.0)))

    def _compute_sdist_by_color(self, agent_xy: np.ndarray, color: str) -> Optional[float]:
        """对该颜色的所有 zone 计算 sdist，取最大值（进入任意一个都为正）。"""
        lst = self._zones_truth.get(color, [])
        if not lst:
            return None
        ax, ay = float(agent_xy[0]), float(agent_xy[1])
        vals: List[float] = []
        for zg in lst:
            if zg.kind == "circle" and zg.radius is not None:
                vals.append(self._sdist_circle(ax, ay, float(zg.center_xy[0]), float(zg.center_xy[1]), float(zg.radius)))
            elif zg.kind == "box" and zg.half_extents is not None:
                vals.append(self._sdist_box(ax, ay, float(zg.center_xy[0]), float(zg.center_xy[1]),
                                            float(zg.half_extents[0]), float(zg.half_extents[1])))
        if not vals:
            return None
        return float(np.max(vals))

    # ----------- Gym 标准接口 -----------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        # 每次 reset 重新定位(防止 wrapper 链/内部实现变化)
        self._introspect_geoms_from_backend()#目的是确保获取到最新的区域信息，因为某些任务在重置后可能会改变区域的位置

        info['propositions'] = set()           # 原环境 cost 推导
        info['propositions_geom'] = set()      # 几何真值推导
        for c in self.colors:
            info[f"sdist_{c}"] = 0.0

        obs['wall_sensor'] = np.array([0, 0, 0, 0], dtype=np.float64)

        # 上电即计算一次几何命题/距离
        agent_xy = self._get_agent_xy()
        info['agent_xy'] = None if agent_xy is None else np.asarray(agent_xy, dtype=float)
        if agent_xy is not None:
            for c in self.colors:
                s = self._compute_sdist_by_color(agent_xy, c)
                if s is not None:
                    info[f"sdist_{c}"] = s
                    if s > 0.0:
                        info['propositions_geom'].add(c)
        else:
            print("agent_xy为空")

        return obs, info

    def step(self, action: ActType):
        obs, reward, cost, terminated, truncated, info = super().step(action)

        if 'wall_sensor' in info:
            obs['wall_sensor'] = info['wall_sensor']

        # 保留原环境的命题（基于 cost_zones_*）
        info['propositions'] = {c for c in self.colors if info.get(f'cost_zones_{c}', 0) > 0}

        # 计算几何命题与带符号边界距
        info['propositions_geom'] = set()
        agent_xy = self._get_agent_xy()
        info['agent_xy'] = None if agent_xy is None else np.asarray(agent_xy, dtype=float)
        if agent_xy is not None:
            for c in self.colors:
                s = self._compute_sdist_by_color(agent_xy, c) #计算几何命题与带符号边界距
                if s is not None:
                    info[f"sdist_{c}"] = s
                    if s > 0.0:
                        info['propositions_geom'].add(c)
                else:
                    info[f"sdist_{c}"] = 0.0  # 保持字段存在
                    print("有向距离为空")
        else:
            for c in self.colors:
                info[f'sdist_{c}'] = 0.0
            print("agent_xy为空")

        # 若任务定义了 ltl 墙体违规，按原逻辑处理
        if 'cost_ltl_walls' in info:
            terminated = terminated or info['cost_ltl_walls'] > 0
            reward = -1.0 if info['cost_ltl_walls'] > 0 else 0.0

        return obs, reward, terminated, truncated, info

    # ----------- Deep-LTL 约定接口 -----------
    def get_propositions(self) -> list[str]:
        return list(self.colors)

    def get_possible_assignments(self) -> list[Assignment]:
        return Assignment.zero_or_one_propositions(set(self.get_propositions()))

    def get_zones_truth(self) -> Dict[str, List[ZoneGeom]]:
        return self._zones_truth


# ================= 自测入口 =================

if __name__ == "__main__":
    import safety_gymnasium

    print(">>> safety_gym_wrapper module loaded, __name__ =", __name__)
    env = safety_gymnasium.make('PointLtl2-v0')
    w = SafetyGymWrapper(env)

    # 打印几何摘要
    print("[TEST] zones truth summary:")
    for c, lst in w.get_zones_truth().items():
        print(f"  {c}: {len(lst)}")
        for z in lst[:5]:
            if z.kind == "circle":
                print("   -", z.name, "(circle)", "center=", z.center_xy, "r=", z.radius)
            elif z.kind == "box":
                print("   -", z.name, "(box)", "center=", z.center_xy, "half_ext=", z.half_extents)
            else:
                print("   -", z.name, f"({z.kind})", "center=", z.center_xy)

    # agent 坐标
    print("[TEST] agent xy:", w._get_agent_xy())

    # reset & 连续 step，观察 sdist / propositions_geom 是否随位置变化
    obs, info = w.reset(seed=0)
    sline0 = {k: _safe_float(info.get(k, None)) for k in info if k.startswith("sdist_")}
    print("[TEST] after reset: propositions_geom:", info.get('propositions_geom', set()))
    print("[TEST] after reset: sdist snapshot:", sline0)

    for t in range(5):
        a = env.action_space.sample()
        obs, rew, ter, tru, inf = w.step(a)
        print("agent位置：",inf["agent_xy"])
        sline = {k: _safe_float(inf.get(k, None)) for k in inf if k.startswith("sdist_")}
        print(f"  step {t}: propositions_geom={inf.get('propositions_geom', set())}  sdist={sline}")
        if ter or tru:
            obs, info = w.reset()
    print("[TEST] done.")
