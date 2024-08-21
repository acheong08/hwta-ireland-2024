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

    def setup(self):
        self.latency_sensitivity = Sensitivity(self.latency_sensitivity)
        return self


@dataclass
class SellingPrices:
    server_generation: ServerGeneration
    latency_sensitivity: Sensitivity
    selling_price: int

    def setup(self):
        self.server_generation = ServerGeneration(self.server_generation)
        self.latency_sensitivity = Sensitivity(self.latency_sensitivity)
        self.selling_price = int(self.selling_price * 10)
        return self


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
        self.server_generation = ServerGeneration(self.server_generation)
        self.server_type = ServerType(self.server_type)
        self.release_time = json.loads(self.release_time)  # type: ignore[reportArgumentType]
        self.purchase_price = int(self.purchase_price * 10)
        return self


@dataclass
class Demand:
    time_step: int
    server_generation: ServerGeneration
    latency_high: int
    latency_medium: int
    latency_low: int

    def setup(self):
        self.server_generation = ServerGeneration(self.server_generation)
        return self

    def get_latency(self, sen: Sensitivity):
        if sen == Sensitivity.HIGH:
            return self.latency_high
        if sen == Sensitivity.MEDIUM:
            return self.latency_medium
        if sen == Sensitivity.LOW:
            return self.latency_low
