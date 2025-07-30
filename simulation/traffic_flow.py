import numpy as np

class VehicleType:
    def __init__(self, id, accel=1.0, decel=4.5, max_speed_kmh=50, length=5):
        self.id = id
        self.accel = accel
        self.decel = decel
        self.max_speed = max_speed_kmh / 3.6  # Convert km/h to m/s
        self.length = length

    def to_xml(self):
        return f'  <vType id="{self.id}" accel="{self.accel}" decel="{self.decel}" maxSpeed="{self.max_speed:.2f}" length="{self.length}"/>'

class TrafficFlow:
    def __init__(self, start_time, end_time, vehicles_per_minute, route_id, edges, vehicle_type):
        self.start_time = start_time
        self.end_time = end_time
        self.vehicles_per_minute = vehicles_per_minute
        self.lambda_rate = vehicles_per_minute / 60
        self.route_id = route_id
        self.edges = edges
        self.vehicle_type = vehicle_type

    def generate_arrivals(self):
        current_time = self.start_time
        arrivals = []

        while current_time < self.end_time:
            inter_arrival = np.random.exponential(1 / self.lambda_rate)
            current_time += inter_arrival
            if current_time >= self.end_time:
                break
            arrivals.append(current_time)

        return arrivals


class TrafficFlowManager:
    def __init__(self, flows):
        self.flows = flows

    def generate_routes_xml(self):
        lines = []
        lines.append('<routes>')
        for flow in self.flows:
            lines.append(flow.vehicle_type.to_xml())

        # Unique IDs for routes
        added_routes = set()
        for flow in self.flows:
            if flow.route_id not in added_routes:
                lines.append(f'  <route id="{flow.route_id}" edges="{flow.edges}"/>')
                added_routes.add(flow.route_id)

        # Vehicles
        car_id = 0
        all_vehicles = []
        for flow in self.flows:
            arrivals = flow.generate_arrivals()
            for t in arrivals:
                all_vehicles.append((t, flow.route_id, car_id, flow.vehicle_type.id))
                car_id += 1

        # Sort for departure time
        all_vehicles.sort(key=lambda x: x[0])

        for t, route_id, cid, type in all_vehicles:
            lines.append(f'  <vehicle id="car{cid}" type="{type}" route="{route_id}" depart="{int(t)}" departSpeed="max"/>')

        lines.append('</routes>')
        return "\n".join(lines)
