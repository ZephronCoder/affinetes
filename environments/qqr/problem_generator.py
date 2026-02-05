"""
Deterministic Problem Generator for Travel Planning Evaluation.

Supports seven problem types:
1. InterCity - Intercity transport planning (required: poi_search + flights/trains + direction + weather)
2. MultiDay - Multi-day trip planning (required: poi_search + around_search + direction + weather)
3. Hybrid - Combined planning (required: all 6 tools)
4. SinglePOI - Deep exploration of a single attraction area
5. FoodTour - Food-focused city exploration
6. Business - Business trip planning with transport
7. FamilyStudy - Educational family trip planning

Core principle: Same task_id always generates the same problem.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from config import (
    MAJOR_CITIES,
    CITY_PAIRS,
    PROBLEM_TYPES,
    REQUIRED_TOOLS_BY_TYPE,
    DIFFICULTY_LEVELS,
)


@dataclass
class TravelProblem:
    """Travel planning problem definition."""

    # Basic info
    task_id: int
    problem_type: str  # intercity, multiday, hybrid, single_poi, food_tour, business, family_study

    # City info
    origin_city: str  # Origin city (InterCity/Hybrid/Business)
    destination_city: str  # Destination city

    # Time info
    travel_date: str  # Departure date YYYY-MM-DD
    num_days: int  # Trip duration (MultiDay/Hybrid)

    # Constraints
    budget: Optional[int] = None  # Budget (CNY/person)
    arrival_constraint: Optional[str] = None  # Arrival time constraint
    preference: Optional[str] = None  # Preference (comfort/economy/speed)
    group_type: str = "solo"  # Group type
    group_size: int = 1

    # Interests and special requirements
    interests: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    # Required tools
    required_tools: List[str] = field(default_factory=list)

    # Difficulty level (1=beginner, 2=intermediate, 3=advanced)
    difficulty: int = 2

    # Budget level for parameterized constraints
    budget_level: str = "economy"  # budget, economy, comfort, luxury

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "problem_type": self.problem_type,
            "origin_city": self.origin_city,
            "destination_city": self.destination_city,
            "travel_date": self.travel_date,
            "num_days": self.num_days,
            "budget": self.budget,
            "arrival_constraint": self.arrival_constraint,
            "preference": self.preference,
            "group_type": self.group_type,
            "group_size": self.group_size,
            "interests": self.interests,
            "constraints": self.constraints,
            "required_tools": self.required_tools,
            "difficulty": self.difficulty,
            "budget_level": self.budget_level,
        }


# ==================== Problem Type Specific Configuration ====================

# InterCity configuration (Chinese content is data, not comments)
INTERCITY_PREFERENCES = ["舒适优先", "经济优先", "速度优先", "无特殊要求"]
INTERCITY_ARRIVAL_CONSTRAINTS = [
    "上午10点前到达", "中午12点前到达", "下午3点前到达",
    "晚上8点前到达", "无时间限制"
]
INTERCITY_BUDGETS = [800, 1000, 1500, 2000, 3000, 5000]

# MultiDay configuration
MULTIDAY_INTERESTS = [
    "自然风光", "文化历史", "美食探店", "购物血拼", "休闲度假",
    "摄影打卡", "亲子游乐", "户外运动", "民俗体验", "博物馆",
    "古镇古村", "温泉养生", "海滨度假", "登山徒步", "赏花观鸟",
]
MULTIDAY_CONSTRAINTS = [
    "避开人群", "预算优先", "轻松休闲", "深度体验",
    "公共交通优先", "无障碍通道", "素食要求", "宠物友好",
    "不乘坐红眼航班", "优先景区免费开放日",
]
MULTIDAY_GROUP_TYPES = ["solo", "couple", "family", "friends", "elderly"]
MULTIDAY_BUDGETS_PER_DAY = [500, 800, 1000, 1500, 2000]
MULTIDAY_DAYS = [2, 3, 4, 5]

# Budget levels with per-day ranges
BUDGET_LEVELS = {
    "budget": (200, 500),
    "economy": (500, 1000),
    "comfort": (1000, 2000),
    "luxury": (2000, 5000),
}

# SinglePOI configuration
SINGLE_POI_TYPES = [
    "故宫", "西湖", "兵马俑", "长城", "外滩", "鼓浪屿",
    "西溪湿地", "太湖", "洱海", "青海湖", "莫高窟", "布达拉宫",
    "张家界国家森林公园", "九寨沟", "黄山", "黄鹤楼", "趵突泉",
    "乐山大佛", "都江堰", "峨眉山",
]

# POI → valid destination cities mapping (POI must be in/near destination)
# Every POI must actually be located in/near its mapped city
POI_CITY_MAP = {
    "故宫": ["北京"],
    "长城": ["北京"],
    "西湖": ["杭州"],
    "西溪湿地": ["杭州"],
    "兵马俑": ["西安"],
    "外滩": ["上海"],
    "鼓浪屿": ["厦门"],
    "太湖": ["苏州", "无锡"],
    "洱海": ["大理"],
    "青海湖": ["西宁"],
    "莫高窟": ["敦煌"],
    "布达拉宫": ["拉萨"],
    "张家界国家森林公园": ["张家界"],
    "九寨沟": ["九寨沟"],
    "黄山": ["黄山"],
    "黄鹤楼": ["武汉"],
    "趵突泉": ["济南"],
    "乐山大佛": ["乐山"],
    "都江堰": ["都江堰", "成都"],
    "峨眉山": ["峨眉山", "乐山"],
}

# FoodTour configuration
FOOD_THEMES = [
    "地方特色小吃", "火锅美食", "海鲜大餐", "早茶文化",
    "夜市美食", "老字号探访", "米其林餐厅", "街头小吃",
    "素食禅意料理", "特色饮品甜品",
]

# Business configuration
BUSINESS_PURPOSES = [
    "商务会议", "客户拜访", "行业展会", "公司年会",
    "商务考察", "项目洽谈",
]

# FamilyStudy configuration
STUDY_THEMES = [
    "科技馆探索", "博物馆研学", "自然科学考察", "历史文化体验",
    "非遗手工体验", "天文观测", "动植物园探索", "古建筑鉴赏",
]


class ProblemGenerator:
    """Deterministic problem generator."""

    def __init__(self):
        self.cities = MAJOR_CITIES
        self.city_pairs = CITY_PAIRS

    def generate(self, task_id: int, difficulty: Optional[int] = None) -> TravelProblem:
        """
        Generate problem deterministically based on task_id.

        task_id determines problem type, cities, dates, and all other parameters.
        difficulty: optional override (1/2/3), default derived from task_id.
        """
        rng = random.Random(task_id)

        # Determine problem type from task_id
        problem_type = PROBLEM_TYPES[task_id % len(PROBLEM_TYPES)]

        # Determine difficulty if not overridden
        if difficulty is None:
            difficulty = (task_id // len(PROBLEM_TYPES)) % 3 + 1

        if problem_type == "intercity":
            return self._generate_intercity(rng, task_id, difficulty)
        elif problem_type == "multiday":
            return self._generate_multiday(rng, task_id, difficulty)
        elif problem_type == "hybrid":
            return self._generate_hybrid(rng, task_id, difficulty)
        elif problem_type == "single_poi":
            return self._generate_single_poi(rng, task_id, difficulty)
        elif problem_type == "food_tour":
            return self._generate_food_tour(rng, task_id, difficulty)
        elif problem_type == "business":
            return self._generate_business(rng, task_id, difficulty)
        else:  # family_study
            return self._generate_family_study(rng, task_id, difficulty)

    def _generate_intercity(self, rng: random.Random, task_id: int, difficulty: int) -> TravelProblem:
        """Generate intercity transport planning problem."""
        # Select city pair
        distance_type = rng.choice(["short", "medium", "long"])
        city_pair = rng.choice(self.city_pairs[distance_type])
        origin, destination = city_pair

        # Generate date based on task_id
        base_date = datetime(2025, 3, 1)
        day_offset = task_id % 365
        travel_date = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")

        # Budget level
        budget_level = rng.choice(list(BUDGET_LEVELS.keys()))
        budget_range = BUDGET_LEVELS[budget_level]
        budget = rng.randint(budget_range[0], budget_range[1])

        # Random constraints
        arrival_constraint = rng.choice(INTERCITY_ARRIVAL_CONSTRAINTS)
        preference = rng.choice(INTERCITY_PREFERENCES)

        # Required tools (transport related)
        required_tools = ["poi_search", "direction", "weather"]
        # Decide main transport mode based on distance
        if distance_type == "short":
            required_tools.append("search_train_tickets")
        elif distance_type == "long":
            required_tools.append("search_flights")
        else:
            # Medium distance: both possible
            required_tools.extend(["search_flights", "search_train_tickets"])

        return TravelProblem(
            task_id=task_id,
            problem_type="intercity",
            origin_city=origin,
            destination_city=destination,
            travel_date=travel_date,
            num_days=1,  # Single day trip
            budget=budget,
            arrival_constraint=arrival_constraint,
            preference=preference,
            required_tools=required_tools,
            difficulty=difficulty,
            budget_level=budget_level,
        )

    def _generate_multiday(self, rng: random.Random, task_id: int, difficulty: int) -> TravelProblem:
        """Generate multi-day trip planning problem."""
        # Select destination city
        destination = rng.choice(self.cities)

        # Generate date
        base_date = datetime(2025, 3, 1)
        day_offset = task_id % 365
        travel_date = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")

        # Trip duration scaled by difficulty
        if difficulty == 1:
            num_days = 2
        elif difficulty == 2:
            num_days = rng.choice([2, 3, 4])
        else:
            num_days = rng.choice([3, 4, 5])

        # Group info
        group_type = rng.choice(MULTIDAY_GROUP_TYPES)
        group_size = self._get_group_size(rng, group_type)

        # Budget
        budget_level = rng.choice(list(BUDGET_LEVELS.keys()))
        budget_range = BUDGET_LEVELS[budget_level]
        budget_per_day = rng.randint(budget_range[0], budget_range[1])
        total_budget = budget_per_day * num_days * group_size

        # Interests (2-4)
        num_interests = rng.randint(2, 4)
        interests = rng.sample(MULTIDAY_INTERESTS, num_interests)

        # Constraints (0-2, more at higher difficulty)
        max_constraints = min(difficulty, 2)
        num_constraints = rng.randint(0, max_constraints)
        constraints = rng.sample(MULTIDAY_CONSTRAINTS, num_constraints) if num_constraints > 0 else []

        # Required tools
        required_tools = REQUIRED_TOOLS_BY_TYPE["multiday"].copy()

        return TravelProblem(
            task_id=task_id,
            problem_type="multiday",
            origin_city="",  # Multi-day trip doesn't need origin city
            destination_city=destination,
            travel_date=travel_date,
            num_days=num_days,
            budget=total_budget,
            group_type=group_type,
            group_size=group_size,
            interests=interests,
            constraints=constraints,
            required_tools=required_tools,
            difficulty=difficulty,
            budget_level=budget_level,
        )

    def _generate_hybrid(self, rng: random.Random, task_id: int, difficulty: int) -> TravelProblem:
        """Generate hybrid planning problem (transport + multi-day itinerary)."""
        # Select city pair (medium/long distance)
        distance_type = rng.choice(["medium", "long"])
        city_pair = rng.choice(self.city_pairs[distance_type])
        origin, destination = city_pair

        # Generate date
        base_date = datetime(2025, 3, 1)
        day_offset = task_id % 365
        travel_date = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")

        # Trip duration (3-5 days)
        num_days = rng.randint(3, 5)

        # Group info
        group_type = rng.choice(MULTIDAY_GROUP_TYPES)
        group_size = self._get_group_size(rng, group_type)

        # Budget (including transport)
        budget_level = rng.choice(list(BUDGET_LEVELS.keys()))
        budget_range = BUDGET_LEVELS[budget_level]
        budget_per_day = rng.randint(budget_range[0], budget_range[1])
        transport_budget = rng.choice([1000, 1500, 2000, 3000])
        total_budget = (budget_per_day * num_days + transport_budget) * group_size

        # Interests
        num_interests = rng.randint(2, 3)
        interests = rng.sample(MULTIDAY_INTERESTS, num_interests)

        # Constraints
        num_constraints = rng.randint(0, 2)
        constraints = rng.sample(MULTIDAY_CONSTRAINTS, num_constraints) if num_constraints > 0 else []

        # Preference
        preference = rng.choice(INTERCITY_PREFERENCES)

        # Required tools (all)
        required_tools = REQUIRED_TOOLS_BY_TYPE["hybrid"].copy()

        return TravelProblem(
            task_id=task_id,
            problem_type="hybrid",
            origin_city=origin,
            destination_city=destination,
            travel_date=travel_date,
            num_days=num_days,
            budget=total_budget,
            preference=preference,
            group_type=group_type,
            group_size=group_size,
            interests=interests,
            constraints=constraints,
            required_tools=required_tools,
            difficulty=difficulty,
            budget_level=budget_level,
        )

    def _generate_single_poi(self, rng: random.Random, task_id: int, difficulty: int) -> TravelProblem:
        """Generate single POI deep exploration problem."""
        poi_focus = rng.choice(SINGLE_POI_TYPES)
        valid_cities = POI_CITY_MAP.get(poi_focus, self.cities)
        destination = rng.choice(valid_cities)

        base_date = datetime(2025, 3, 1)
        day_offset = task_id % 365
        travel_date = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")

        group_type = rng.choice(MULTIDAY_GROUP_TYPES)
        group_size = self._get_group_size(rng, group_type)

        budget_level = rng.choice(["budget", "economy", "comfort"])
        budget_range = BUDGET_LEVELS[budget_level]
        budget = rng.randint(budget_range[0], budget_range[1]) * group_size

        interests = [poi_focus, rng.choice(["摄影打卡", "深度体验", "历史文化", "自然风光"])]
        constraints = []
        if difficulty >= 2:
            constraints = rng.sample(["避开人群", "无障碍通道", "轻松休闲"], rng.randint(0, 1))

        required_tools = REQUIRED_TOOLS_BY_TYPE["single_poi"].copy()

        return TravelProblem(
            task_id=task_id,
            problem_type="single_poi",
            origin_city="",
            destination_city=destination,
            travel_date=travel_date,
            num_days=1,
            budget=budget,
            group_type=group_type,
            group_size=group_size,
            interests=interests,
            constraints=constraints,
            required_tools=required_tools,
            difficulty=difficulty,
            budget_level=budget_level,
        )

    def _generate_food_tour(self, rng: random.Random, task_id: int, difficulty: int) -> TravelProblem:
        """Generate food-focused tour problem."""
        # Pick cities known for food
        food_cities = [c for c in self.cities if c in [
            "成都", "重庆", "广州", "西安", "长沙", "武汉", "北京", "上海",
            "南京", "杭州", "厦门", "昆明", "贵阳", "福州", "泉州", "洛阳",
        ]]
        if not food_cities:
            food_cities = self.cities
        destination = rng.choice(food_cities)

        base_date = datetime(2025, 3, 1)
        day_offset = task_id % 365
        travel_date = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")

        num_days = rng.choice([1, 2]) if difficulty <= 2 else rng.choice([2, 3])

        group_type = rng.choice(["solo", "couple", "friends"])
        group_size = self._get_group_size(rng, group_type)

        budget_level = rng.choice(list(BUDGET_LEVELS.keys()))
        budget_range = BUDGET_LEVELS[budget_level]
        budget = rng.randint(budget_range[0], budget_range[1]) * num_days * group_size

        food_theme = rng.choice(FOOD_THEMES)
        interests = ["美食探店", food_theme]
        if difficulty >= 2:
            interests.append(rng.choice(["摄影打卡", "民俗体验", "购物血拼"]))

        constraints = []
        if rng.random() < 0.3:
            constraints.append(rng.choice(["素食要求", "预算优先", "避开人群"]))

        required_tools = REQUIRED_TOOLS_BY_TYPE["food_tour"].copy()

        return TravelProblem(
            task_id=task_id,
            problem_type="food_tour",
            origin_city="",
            destination_city=destination,
            travel_date=travel_date,
            num_days=num_days,
            budget=budget,
            group_type=group_type,
            group_size=group_size,
            interests=interests,
            constraints=constraints,
            required_tools=required_tools,
            difficulty=difficulty,
            budget_level=budget_level,
        )

    def _generate_business(self, rng: random.Random, task_id: int, difficulty: int) -> TravelProblem:
        """Generate business trip problem."""
        distance_type = rng.choice(["medium", "long"])
        city_pair = rng.choice(self.city_pairs[distance_type])
        origin, destination = city_pair

        base_date = datetime(2025, 3, 1)
        day_offset = task_id % 365
        travel_date = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")

        num_days = rng.choice([1, 2]) if difficulty <= 2 else rng.choice([2, 3])

        purpose = rng.choice(BUSINESS_PURPOSES)
        preference = rng.choice(["舒适优先", "速度优先"])
        arrival_constraint = rng.choice(["上午10点前到达", "中午12点前到达", "无时间限制"])

        budget_level = rng.choice(["comfort", "luxury"])
        budget_range = BUDGET_LEVELS[budget_level]
        budget = rng.randint(budget_range[0], budget_range[1]) * num_days

        interests = [purpose]
        constraints = []
        if difficulty >= 2:
            constraints.append(rng.choice(["不乘坐红眼航班", "公共交通优先"]))

        required_tools = REQUIRED_TOOLS_BY_TYPE["business"].copy()

        return TravelProblem(
            task_id=task_id,
            problem_type="business",
            origin_city=origin,
            destination_city=destination,
            travel_date=travel_date,
            num_days=num_days,
            budget=budget,
            arrival_constraint=arrival_constraint,
            preference=preference,
            interests=interests,
            constraints=constraints,
            required_tools=required_tools,
            difficulty=difficulty,
            budget_level=budget_level,
        )

    def _generate_family_study(self, rng: random.Random, task_id: int, difficulty: int) -> TravelProblem:
        """Generate family educational trip problem."""
        destination = rng.choice(self.cities)

        base_date = datetime(2025, 3, 1)
        day_offset = task_id % 365
        travel_date = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")

        num_days = rng.choice([2, 3]) if difficulty <= 2 else rng.choice([3, 4, 5])

        group_type = "family"
        group_size = rng.randint(3, 5)

        budget_level = rng.choice(["economy", "comfort"])
        budget_range = BUDGET_LEVELS[budget_level]
        budget = rng.randint(budget_range[0], budget_range[1]) * num_days * group_size

        study_theme = rng.choice(STUDY_THEMES)
        interests = [study_theme, "亲子游乐"]
        if difficulty >= 2:
            interests.append(rng.choice(["文化历史", "自然风光", "博物馆"]))

        constraints = ["无障碍通道"]
        if difficulty >= 3:
            constraints.append(rng.choice(["轻松休闲", "预算优先"]))

        required_tools = REQUIRED_TOOLS_BY_TYPE["family_study"].copy()

        return TravelProblem(
            task_id=task_id,
            problem_type="family_study",
            origin_city="",
            destination_city=destination,
            travel_date=travel_date,
            num_days=num_days,
            budget=budget,
            group_type=group_type,
            group_size=group_size,
            interests=interests,
            constraints=constraints,
            required_tools=required_tools,
            difficulty=difficulty,
            budget_level=budget_level,
        )

    def _get_group_size(self, rng: random.Random, group_type: str) -> int:
        """Determine group size based on group type."""
        if group_type == "solo":
            return 1
        elif group_type == "couple":
            return 2
        elif group_type == "family":
            return rng.randint(3, 5)
        elif group_type == "friends":
            return rng.randint(3, 6)
        elif group_type == "elderly":
            return rng.randint(2, 4)
        return 2

    def to_prompt(self, problem: TravelProblem) -> str:
        """Convert problem to natural language prompt."""
        if problem.problem_type == "intercity":
            return self._intercity_to_prompt(problem)
        elif problem.problem_type == "multiday":
            return self._multiday_to_prompt(problem)
        elif problem.problem_type == "hybrid":
            return self._hybrid_to_prompt(problem)
        elif problem.problem_type == "single_poi":
            return self._single_poi_to_prompt(problem)
        elif problem.problem_type == "food_tour":
            return self._food_tour_to_prompt(problem)
        elif problem.problem_type == "business":
            return self._business_to_prompt(problem)
        else:
            return self._family_study_to_prompt(problem)

    def _intercity_to_prompt(self, problem: TravelProblem) -> str:
        """Generate prompt for intercity transport planning problem."""
        parts = [f"我计划从{problem.origin_city}去{problem.destination_city}"]

        if problem.travel_date:
            parts.append(f"出发日期是{problem.travel_date}")

        if problem.budget:
            parts.append(f"预算{problem.budget}元/人")

        if problem.arrival_constraint and problem.arrival_constraint != "无时间限制":
            parts.append(f"希望{problem.arrival_constraint}")

        if problem.preference and problem.preference != "无特殊要求":
            parts.append(f"偏好{problem.preference}")

        prompt = "，".join(parts) + "。"
        prompt += "\n\n请帮我：\n"
        prompt += "1. 查询可选的交通方式（航班/火车）\n"
        prompt += "2. 推荐最佳出行方案并说明理由\n"
        prompt += "3. 提供到达目的地后的简要安排建议"

        return prompt

    def _multiday_to_prompt(self, problem: TravelProblem) -> str:
        """Generate prompt for multi-day trip planning problem."""
        # Group description
        group_desc = self._get_group_description(problem)

        parts = [f"请为我规划一次{problem.destination_city}{problem.num_days}日游"]

        parts.append(f"出行人员：{group_desc}")

        if problem.travel_date:
            parts.append(f"出发日期：{problem.travel_date}")

        if problem.budget:
            parts.append(f"总预算：{problem.budget}元")

        prompt = "，".join(parts) + "。"

        if problem.interests:
            prompt += f"\n\n兴趣偏好：{', '.join(problem.interests)}"

        if problem.constraints:
            prompt += f"\n特殊要求：{', '.join(problem.constraints)}"

        prompt += "\n\n请提供详细的每日行程安排，包括：\n"
        prompt += "1. 每天的景点安排和路线规划\n"
        prompt += "2. 餐饮和住宿建议\n"
        prompt += "3. 交通方式和预计时间\n"
        prompt += "4. 预计花费明细"

        return prompt

    def _hybrid_to_prompt(self, problem: TravelProblem) -> str:
        """Generate prompt for hybrid planning problem."""
        group_desc = self._get_group_description(problem)

        prompt = f"我计划从{problem.origin_city}出发去{problem.destination_city}玩{problem.num_days}天。\n\n"

        prompt += f"基本信息：\n"
        prompt += f"- 出行人员：{group_desc}\n"
        prompt += f"- 出发日期：{problem.travel_date}\n"
        prompt += f"- 总预算：{problem.budget}元\n"

        if problem.preference and problem.preference != "无特殊要求":
            prompt += f"- 交通偏好：{problem.preference}\n"

        if problem.interests:
            prompt += f"\n兴趣偏好：{', '.join(problem.interests)}"

        if problem.constraints:
            prompt += f"\n特殊要求：{', '.join(problem.constraints)}"

        prompt += "\n\n请帮我完成完整的旅行规划：\n"
        prompt += "1. 往返交通方案（航班/火车选择和推荐）\n"
        prompt += "2. 每日详细行程安排\n"
        prompt += "3. 餐饮和住宿建议\n"
        prompt += "4. 完整预算明细"

        return prompt

    def _single_poi_to_prompt(self, problem: TravelProblem) -> str:
        """Generate prompt for single POI deep exploration."""
        group_desc = self._get_group_description(problem)
        poi_name = problem.interests[0] if problem.interests else "景点"

        prompt = f"我想在{problem.destination_city}深度游览{poi_name}及其周边区域。\n\n"
        prompt += f"基本信息：\n"
        prompt += f"- 出行人员：{group_desc}\n"
        prompt += f"- 日期：{problem.travel_date}\n"
        if problem.budget:
            prompt += f"- 预算：{problem.budget}元\n"

        if len(problem.interests) > 1:
            prompt += f"\n重点关注：{', '.join(problem.interests[1:])}"

        if problem.constraints:
            prompt += f"\n特殊要求：{', '.join(problem.constraints)}"

        prompt += "\n\n请帮我规划：\n"
        prompt += f"1. {poi_name}的最佳游览路线和时间安排\n"
        prompt += "2. 周边值得一去的景点、餐厅、咖啡馆\n"
        prompt += "3. 交通方式和各景点间的距离\n"
        prompt += "4. 实用小贴士（开放时间、门票、注意事项等）"

        return prompt

    def _food_tour_to_prompt(self, problem: TravelProblem) -> str:
        """Generate prompt for food tour."""
        group_desc = self._get_group_description(problem)
        food_theme = problem.interests[1] if len(problem.interests) > 1 else "当地美食"

        prompt = f"我想在{problem.destination_city}来一次{food_theme}之旅"
        if problem.num_days > 1:
            prompt += f"，计划{problem.num_days}天"
        prompt += "。\n\n"

        prompt += f"基本信息：\n"
        prompt += f"- 出行人员：{group_desc}\n"
        prompt += f"- 日期：{problem.travel_date}\n"
        if problem.budget:
            prompt += f"- 餐饮预算：{problem.budget}元\n"

        if problem.constraints:
            prompt += f"\n特殊要求：{', '.join(problem.constraints)}"

        prompt += "\n\n请帮我规划：\n"
        prompt += "1. 必吃的特色美食和推荐店铺\n"
        prompt += "2. 美食路线规划（按区域或时间）\n"
        prompt += "3. 各店铺之间的交通方式和距离\n"
        prompt += "4. 预计花费和用餐建议"

        return prompt

    def _business_to_prompt(self, problem: TravelProblem) -> str:
        """Generate prompt for business trip."""
        purpose = problem.interests[0] if problem.interests else "商务出行"

        prompt = f"我因{purpose}需要从{problem.origin_city}前往{problem.destination_city}"
        if problem.num_days > 1:
            prompt += f"，预计{problem.num_days}天"
        prompt += "。\n\n"

        prompt += f"基本信息：\n"
        prompt += f"- 出发日期：{problem.travel_date}\n"
        if problem.budget:
            prompt += f"- 差旅预算：{problem.budget}元\n"
        if problem.preference:
            prompt += f"- 交通偏好：{problem.preference}\n"
        if problem.arrival_constraint and problem.arrival_constraint != "无时间限制":
            prompt += f"- 到达要求：{problem.arrival_constraint}\n"

        if problem.constraints:
            prompt += f"\n特殊要求：{', '.join(problem.constraints)}"

        prompt += "\n\n请帮我规划：\n"
        prompt += "1. 最优交通方案（航班/高铁）\n"
        prompt += "2. 商务区附近的酒店推荐\n"
        prompt += "3. 周边餐饮和商务配套\n"
        prompt += "4. 差旅费用预估"

        return prompt

    def _family_study_to_prompt(self, problem: TravelProblem) -> str:
        """Generate prompt for family educational trip."""
        study_theme = problem.interests[0] if problem.interests else "亲子研学"

        prompt = f"我们一家{problem.group_size}口想去{problem.destination_city}进行一次{study_theme}之旅"
        prompt += f"，计划{problem.num_days}天。\n\n"

        prompt += f"基本信息：\n"
        prompt += f"- 出发日期：{problem.travel_date}\n"
        if problem.budget:
            prompt += f"- 总预算：{problem.budget}元\n"

        if problem.interests:
            prompt += f"\n学习主题：{', '.join(problem.interests)}"

        if problem.constraints:
            prompt += f"\n特殊要求：{', '.join(problem.constraints)}"

        prompt += "\n\n请帮我规划：\n"
        prompt += "1. 适合亲子的景点和学习活动\n"
        prompt += "2. 每日行程安排（注意儿童体力）\n"
        prompt += "3. 亲子友好的餐厅和住宿\n"
        prompt += "4. 教育意义和互动体验建议"

        return prompt

    def _get_group_description(self, problem: TravelProblem) -> str:
        """Get group description in Chinese."""
        desc_map = {
            "solo": "独自一人",
            "couple": "情侣二人",
            "family": f"{problem.group_size}口之家",
            "friends": f"{problem.group_size}位朋友结伴",
            "elderly": f"{problem.group_size}位老年人",
        }
        return desc_map.get(problem.group_type, f"{problem.group_size}人")


# Singleton
_default_generator = None


def get_generator() -> ProblemGenerator:
    """Get default generator instance."""
    global _default_generator
    if _default_generator is None:
        _default_generator = ProblemGenerator()
    return _default_generator
