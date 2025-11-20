import numpy as np                              # 导入 numpy，用于数值和数组处理
import enum                                      # 导入 enum，用于定义枚举类型（区域颜色）
import gym                                       # 导入 OpenAI Gym，用于 observation/action 空间定义
import random                                    # 导入 random，可能用于随机选择（尽管这段代码并未显式用到）

from safety_gym.envs.engine import Engine       # 从 safety_gym 导入 Engine，所有环境继承自它，封装了 mujoco 相关逻辑

# 定义 zone 枚举，表示不同颜色/类型的区域
class zone(enum.Enum):
    JetBlack = 0                                 # JetBlack 区域，值为 0
    White    = 1                                 # White 区域，值为 1
    Blue     = 2                                 # Blue 区域，值为 2
    Green    = 3                                 # Green 区域，值为 3
    Red      = 4                                 # Red 区域，值为 4
    Yellow   = 5                                 # Yellow 区域，值为 5
    Cyan     = 6                                 # Cyan 区域，值为 6
    Magenta  = 7                                 # Magenta 区域，值为 7

    def __lt__(self, sth):                        # 定义小于比较，用于对 zone 列表排序
        return self.value < sth.value            # 比较枚举值大小

    def __str__(self):                           # 定义 str() 输出行为
        return self.name[0]                      # 返回枚举名称的第一个字符（例如 'Red' -> 'R'）

    def __repr__(self):                          # 定义 repr() 输出行为（调试或打印列表时用）
        return self.name                         # 返回枚举的完整名字（例如 'Red'）

GROUP_ZONE = 7                                   # MuJoCo 中用于标记 zone 的碰撞/渲染组编号（任意整数常量）

# 定义主环境类 ZonesEnv，继承 safety_gym 的 Engine（包含大量构建世界、观测、lidar 等逻辑）
class ZonesEnv(Engine):
    """
    This environment is a modification of the Safety-Gym's environment.
    There is no "goal circle" but rather a collection of zones that the
    agent has to visit or to avoid in order to finish the task.

    For now we only support the 'point' robot.
    """
    def __init__(self, zones:list, use_fixed_map:float, timeout:int, config:dict, walled=True, map_seed=None):
        # 更新默认配置（Engine 有 DEFAULT 字典，这里扩展 zone 相关的默认参数）
        self.DEFAULT.update({
            'observe_zones': False,            # 是否在观测中包含 zones 的 lidar 信息（默认否）
            'zones_num': 0,                    # 初始化 zones 数量（会被后续覆盖）
            'zones_placements': None,          # zones 的放置列表（如果 None 则使用默认 placement 机制）
            'zones_locations': [],             # 明确给定的 zones 位置列表（覆盖 placements）
            'zones_keepout': 0.55,             # 放置时的保持距离（防止太靠近）
            'zones_size': 0.25,                # zone 的半径/大小
        })
        if (walled):                          # 如果需要围墙（walled=True）
            world_extent = 2.5               # 世界边界半宽
            # 生成沿四边的墙的位置（这里构造一个非常细的网格的墙坐标列表）
            walls = [(i/10, j) for i in range(int(-world_extent * 10),int(world_extent * 10 + 1),1) for j in [-world_extent, world_extent]]
            walls += [(i, j/10) for i in [-world_extent, world_extent] for j in range(int(-world_extent * 10), int(world_extent * 10 + 1),1)]
            # 更新 DEFAULT 中与墙相关的放置/尺寸信息
            self.DEFAULT.update({
                'placements_extents': [-world_extent, -world_extent, world_extent, world_extent],  # 放置搜索范围（x_min,y_min,x_max,y_max）
                'walls_num': len(walls),         # 墙的数量
                'walls_locations': walls,        # 墙的位置列表
                'walls_size': 0.1,               # 墙的厚度/大小
            })

        # 保存传入的 zones（一个 zone 枚举列表，指定每个 zone 的颜色/类型）
        self.zones = zones
        # 获取不同的 zone 类型集合（去重）
        self.zone_types = list(set(zones))
        self.zone_types.sort()                # 对 zone_types 排序（使用上面定义的 __lt__）
        self.use_fixed_map = use_fixed_map    # 是否使用固定地图（评估时通常为 True）
        # 定义每个 zone 类型对应的 RGBA 颜色（用于渲染）
        self._rgb = {
            zone.JetBlack: [0, 0, 0, 1],
            zone.Blue    : [0, 0, 1, 1],
            zone.Green   : [0, 1, 0, 1],
            zone.Cyan    : [0, 1, 1, 1],
            zone.Red     : [1, 0, 0, 1],
            zone.Magenta : [1, 0, 1, 1],
            zone.Yellow  : [1, 1, 0, 1],
            zone.White   : [1, 1, 1, 1]
        }
        # 为每个具体 zone（位置有序对应 self.zones）生成 RGBA 数组（顺序与 self.zones 一致）
        self.zone_rgbs = np.array([self._rgb[haz] for haz in self.zones])
        self.map_seed = map_seed              # 固定地图时使用的随机种子（可选）

        # 构造传给父类 Engine 的配置字典（指定 robot xml、lidar 数量、zones 数量、步长上限等）
        parent_config = {
            'robot_base': 'xmls/point.xml',   # 使用 point 机器人模型
            'task': 'none',                   # 不使用 safety_gym 的内置任务
            'lidar_num_bins': 16,             # lidar 分辨率（测距 ray 数）
            'observe_zones': True,            # 在父类中启用 observe_zones（默认由 DEFAULT 控制）
            'zones_num': len(zones),          # zones 数量（传入的 zones 列表长度）
            'num_steps': timeout              # episode 最大步长（timeout）
        }
        parent_config.update(config)         # 覆盖/合并外部传入的 config（如果传了）

        super().__init__(parent_config)      # 调用父类构造函数，初始化 Engine（建立 mujoco world 等）

    @property
    def zones_pos(self):
        ''' Helper to get the zones positions from layout '''
        # 读取 mujoco 中每个 zone 的 body 的世界坐标位置（通过 self.data.get_body_xpos）
        return [self.data.get_body_xpos(f'zone{i}').copy() for i in range(self.zones_num)]

    def build_observation_space(self):
        # 先调用父类实现，父类会填充 self.obs_space_dict 的基础观测项（例如 lidar、robot 状态等）
        super().build_observation_space()

        if self.observe_zones:               # 如果配置为观测 zones（self.DEFAULT/parent_config 应当设置）
            # 为每种 zone 类型添加一个独立的 lidar 观测条目
            for zone_type in self.zone_types:
                # 在 obs_space_dict 中加入形如 'zones_lidar_Red': Box(0,1,(lidar_bins,))
                self.obs_space_dict.update({f'zones_lidar_{zone_type}': gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)})

        if self.observation_flatten:         # 如果要求扁平化观测（将 dict 展平成向量）
            # 计算扁平后向量的长度
            self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
            # 定义 observation_space 为一个 Box 向量空间
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        else:
            # 否则 observation_space 就是一个 Dict，包含所有子观测空间
            self.observation_space = gym.spaces.Dict(self.obs_space_dict)

    def build_placements_dict(self):
        # 调用父类以建立 placements 的基础字典（父类会读取 DEFAULT 中的 placements_extents 等）
        super().build_placements_dict()

        if self.zones_num:                   # 如果有 zones 要放置
            # 使用 Engine 提供的 helper（placements_dict_from_object）把 zone 的 placement 条目加入到 self.placements
            self.placements.update(self.placements_dict_from_object('zone'))

    def build_world_config(self):
        # 从父类获取基础 world_config（包括 robot、墙、地面等）
        world_config = super().build_world_config()
        # 为每个 zone 创建一个 geometry 描述，并加入 world_config['geoms']
        for i in range(self.zones_num):
            name = f'zone{i}'                # zone 的名字，例如 'zone0','zone1'
            geom = {'name': name,
                    'size': [self.zones_size, 1e-2],      # zone 几何体尺寸，第一个元素为半径，第二个为高度（很薄）
                    'pos': np.r_[self.layout[name], 2e-2],# zone 放置位置，layout 由父类 placements 决定，这里把 z 高度设置为 0.02
                    'rot': self.random_rot(),             # 随机旋转（圆柱体中旋转并不重要）
                    'type': 'cylinder',                   # 类型为 cylinder（圆柱）
                    'contype': 0,                         # 碰撞相关类型，0 表示不参与物理接触（或根据父类语义）
                    'conaffinity': 0,                     # 碰撞亲和性（与 contype 配套）
                    'group': GROUP_ZONE,                  # 指定碰撞/渲染组
                    'rgba': self.zone_rgbs[i] * [1, 1, 1, 0.25]} # 透明化的颜色（最后一个值 alpha = 0.25）
            # 将该 geom 放入 world_config 的 geoms 字典中
            world_config['geoms'][name] = geom

        return world_config                    # 返回完整 world_config 给父类用于生成 xml/场景

    def build_obs(self):
        # 先获取父类已经构造好的观测（包含 basic lidar、robot 状态等）
        obs = super().build_obs()

        if self.observe_zones:                  # 如果需要在观测中加入 zones 的 lidar 信息
            for zone_type in self.zone_types:
                # 找到属于该类型的 zone 的索引列表
                ind = [i for i, z in enumerate(self.zones) if (self.zones[i] == zone_type)]
                # 根据索引从 zones_pos（位置列表）中抽取该类型的位置信息
                pos_in_type = list(np.array(self.zones_pos)[ind])

                # 生成该类型的 lidar 观测（obs_lidar 接收位置列表和组号），并写入 obs 字典
                obs[f'zones_lidar_{zone_type}'] = self.obs_lidar(pos_in_type, GROUP_ZONE)

        return obs                              # 返回完整观测字典

    def render_lidars(self):
        # 调用父类渲染 lidar 的基础行为，得到当前渲染偏移（用于放置多个 lidar 可视化）
        offset = super().render_lidars()

        if self.render_lidar_markers:           # 如果被配置为渲染 lidar 标记（可视化）
            for zone_type in self.zone_types:
                # 只有当 obs_space_dict 中存在该 zone 的 lidar 项时才渲染（防护）
                if f'zones_lidar_{zone_type}' in self.obs_space_dict:
                    # 找到属于该颜色的 zones 的索引
                    ind = [i for i, z in enumerate(self.zones) if (self.zones[i] == zone_type)]
                    # 取出这些 zone 的位置
                    pos_in_type = list(np.array(self.zones_pos)[ind])

                    # 调用父类的 render_lidar，给这些点渲染可视化（颜色使用 _rgb 映射）
                    self.render_lidar(pos_in_type, np.array([self._rgb[zone_type]]), offset, GROUP_ZONE)
                    # 更新 offset，以便下一个类型的 lidar 在画面上不重叠
                    offset += self.render_lidar_offset_delta

        return offset                            # 返回最新的 offset

    # NOTE: self.map_seed is only used for a fixed map during evaluation
    # reset() will not affect the map at all. If we need a new map, create 
    # a new env from scratch with different seed
    def seed(self, seed=None):
        # 如果使用固定地图，则把内部种子 _seed 设置为 map_seed（如果调用 seed() 时没有提供 seed，则用 map_seed；若提供 seed，则不改变）
        if self.use_fixed_map:
            self._seed = self.map_seed if seed is None else None


# LTLZonesEnv 进一步继承 ZonesEnv，提供 LTL 相关的接口（命题 & 事件）
class LTLZonesEnv(ZonesEnv):
    def __init__(self, zones:list, use_fixed_map:float, timeout:int, config={}, map_seed=None):
        # 直接把参数透传给父类构造函数
        super().__init__(zones=zones, use_fixed_map=use_fixed_map, timeout=timeout, config=config, map_seed=map_seed)

    def get_propositions(self):
        # 返回环境中存在的命题集合（将 zone_types 转成字符串列表）
        return [str(i) for i in self.zone_types]

    def get_events(self):
        # 返回当前时刻触发的事件（agent 所在 zone 的名称字符）
        events = ""
        # 遍历每个 zone 的位置
        for h_inedx, h_pos in enumerate(self.zones_pos):
            h_dist = self.dist_xy(h_pos)        # 计算 agent 与该 zone 在 XY 平面上的距离（Engine 提供 dist_xy）
            if h_dist <= self.zones_size:       # 如果距离小于等于 zone 半径，则视为进入该 zone
                # We assume the agent to be in one zone at a time
                events += str(self.zones[h_inedx])  # 把该 zone 的名字（枚举转字符串）追加到事件字符串中

        return events                          # 返回事件字符串（可能是空字符串或单个字符等）

# 以下是一些具体的环境预设类，方便直接创建常用配置
class ZonesEnv1(LTLZonesEnv):
    def __init__(self):
        # 只有一个 Red zone，随机地图（use_fixed_map=False），1000 步的 timeout
        super().__init__(zones=[zone.Red], use_fixed_map=False, timeout=1000)

class ZonesEnv1Fixed(LTLZonesEnv):
    def __init__(self):
        config = {
            # 可以在这里覆盖 placements_extents 等配置（当前注释掉）
            # 'placements_extents': [-1.5, -1.5, 1.5, 1.5]
        }
        # 单个 Red zone，但使用固定地图（use_fixed_map=True）
        super().__init__(zones=[zone.Red], use_fixed_map=True, timeout=1000, config=config)

class ZonesEnv8(LTLZonesEnv):
    def __init__(self, timeout=1000):
        # 八个 zone（两两重复颜色），使用固定地图
        super().__init__(zones=[zone.JetBlack, zone.JetBlack, zone.Red, zone.Red, zone.White, zone.White,  zone.Yellow, zone.Yellow], use_fixed_map=True, timeout=timeout)
        
class ZonesEnv8Fixed(LTLZonesEnv):
    def __init__(self, map_seed, timeout=1000):
        # 同上，但允许指定 map_seed（用于评估时重复地图）
        super().__init__(zones=[zone.JetBlack, zone.JetBlack, zone.Red, zone.Red, zone.White, zone.White,  zone.Yellow, zone.Yellow], use_fixed_map=True, timeout=timeout, map_seed=map_seed)
