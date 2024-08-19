import json
from dataclasses import dataclass
from enum import Enum


class ServerGeneration(Enum):
    CPU_S1 = "CPU.S1"
    CPU_S2 = "CPU.S2"
    CPU_S3 = "CPU.S3"
    CPU_S4 = "CPU.S4"
    GPU_S1 = "GPU.S1"
    GPU_S2 = "GPU.S2"
    GPU_S3 = "GPU.S3"


@dataclass
class Demand:
    time_step: int
    server_generation: ServerGeneration
    latency_high: int
    latency_medium: int
    latency_low: int


class Sensitivity(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Datacenter:
    datacenter_id: str
    cost_of_energy: float
    latency_sensitivity: Sensitivity
    slots_capacity: int


@dataclass
class SellingPrices:
    server_generation: ServerGeneration
    latency_sensitivity: Sensitivity
    selling_price: int


class ServerType(Enum):
    GPU = "GPU"
    CPU = "CPU"


@dataclass
class Server:
    server_generation: ServerGeneration
    server_type: ServerType
    release_time: list[int]
    purchase_price: int
    slots_size: int
    energy_consumption: int
    capacity: int
    life_expectancy: int
    cost_of_moving: int
    average_maintenance_fee: int

    def setup(self):
        self.release_time = json.loads(self.release_time)  # type: ignore[reportArgumentType]
        return self
