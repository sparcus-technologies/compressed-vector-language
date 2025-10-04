import json
import random
import time
from typing import Any, Dict, List

import numpy as np


class RealAgentDataGenerator:
    """Generate realistic agent communication data"""

    def __init__(self):
        # Real agent communication patterns
        self.navigation_commands = [
            "move to coordinates {x}, {y}",
            "navigate to waypoint {waypoint}",
            "proceed to target location {target}",
            "adjust heading to {bearing} degrees",
            "maintain altitude {altitude} meters",
            "reduce speed to {speed} km/h",
            "hold position at current coordinates",
            "return to base station",
            "follow planned route segment {segment}",
            "execute evasive maneuver pattern {pattern}",
        ]

        self.status_updates = [
            "battery level at {battery}%",
            "fuel remaining: {fuel} liters",
            "system diagnostics: all green",
            "temperature readings nominal",
            "GPS signal strength: {signal} bars",
            "sensor array functioning normally",
            "communication link established",
            "mission progress: {progress}% complete",
            "payload status: {payload_status}",
            "estimated time to completion: {eta} minutes",
        ]

        self.obstacle_reports = [
            "obstacle detected at bearing {bearing} degrees",
            "static obstruction ahead at {distance} meters",
            "moving object tracked at coordinates {x}, {y}",
            "weather conditions deteriorating",
            "no-fly zone boundary approached",
            "terrain elevation warning",
            "collision avoidance system activated",
            "visual obstruction due to {condition}",
            "electromagnetic interference detected",
            "restricted airspace entered",
        ]

        self.coordination_messages = [
            "forming up in {formation} formation",
            "maintaining separation distance {distance} meters",
            "synchronizing with agent {agent_id}",
            "requesting permission to {action}",
            "acknowledging command from {commander}",
            "standby for further instructions",
            "mission parameter update received",
            "coordinating approach with team alpha",
            "establishing communication relay",
            "requesting backup from nearest unit",
        ]

        self.emergency_alerts = [
            "emergency landing required immediately",
            "critical system failure detected",
            "requesting immediate assistance",
            "abort mission - return to base",
            "medical emergency - priority override",
            "hostile contact at coordinates {x}, {y}",
            "equipment malfunction - cannot continue",
            "weather emergency - seeking shelter",
            "communication blackout imminent",
            "fuel emergency - immediate landing required",
        ]

        self.locations = [
            "alpha",
            "bravo",
            "charlie",
            "delta",
            "echo",
            "foxtrot",
            "base_1",
            "checkpoint_north",
            "landing_zone_2",
            "rally_point",
        ]

        self.formations = [
            "diamond",
            "wedge",
            "line_abreast",
            "column",
            "echelon",
            "finger_four",
            "combat_spread",
            "trail",
        ]

    def generate_message(self, message_type: str = None) -> Dict[str, Any]:
        """Generate a realistic agent message"""
        if message_type is None:
            message_type = random.choice(
                ["navigation", "status", "obstacle", "coordination", "emergency"]
            )

        # Base message structure
        message = {
            "timestamp": time.time(),
            "agent_id": f"agent_{random.randint(1, 100):03d}",
            "message_type": message_type,
            "priority": random.choice(["low", "normal", "high", "critical"]),
            "mission_id": f"mission_{random.randint(1, 20):02d}",
        }

        # Generate type-specific content
        if message_type == "navigation":
            template = random.choice(self.navigation_commands)
            content = template.format(
                x=round(random.uniform(-1000, 1000), 1),
                y=round(random.uniform(-1000, 1000), 1),
                waypoint=random.choice(self.locations),
                target=random.choice(self.locations),
                bearing=random.randint(0, 359),
                altitude=random.randint(50, 500),
                speed=random.randint(10, 100),
                segment=random.randint(1, 10),
                pattern=random.choice(["alpha", "beta", "gamma"]),
            )

        elif message_type == "status":
            template = random.choice(self.status_updates)
            content = template.format(
                battery=random.randint(10, 100),
                fuel=round(random.uniform(5, 50), 1),
                signal=random.randint(1, 5),
                progress=random.randint(0, 100),
                payload_status=random.choice(["secure", "deployed", "loading"]),
                eta=random.randint(5, 120),
            )

        elif message_type == "obstacle":
            template = random.choice(self.obstacle_reports)
            content = template.format(
                bearing=random.randint(0, 359),
                distance=random.randint(10, 1000),
                x=round(random.uniform(-1000, 1000), 1),
                y=round(random.uniform(-1000, 1000), 1),
                condition=random.choice(["fog", "rain", "dust", "smoke"]),
            )

        elif message_type == "coordination":
            template = random.choice(self.coordination_messages)
            content = template.format(
                formation=random.choice(self.formations),
                distance=random.randint(50, 500),
                agent_id=f"agent_{random.randint(1, 100):03d}",
                action=random.choice(["land", "takeoff", "patrol", "investigate"]),
                commander=f"control_{random.randint(1, 5)}",
            )

        elif message_type == "emergency":
            template = random.choice(self.emergency_alerts)
            content = template.format(
                x=round(random.uniform(-1000, 1000), 1),
                y=round(random.uniform(-1000, 1000), 1),
            )
            message["priority"] = "critical"

        message["content"] = content
        message["content_length"] = len(content)

        return message

    def generate_dataset(self, num_messages: int = 1000) -> List[Dict[str, Any]]:
        """Generate a dataset of realistic agent messages"""
        dataset = []

        # Distribution of message types (realistic)
        message_types = (
            ["navigation"] * 300  # 30% navigation
            + ["status"] * 250  # 25% status updates
            + ["coordination"] * 200  # 20% coordination
            + ["obstacle"] * 150  # 15% obstacle reports
            + ["emergency"] * 100  # 10% emergency
        )

        for i in range(num_messages):
            msg_type = random.choice(message_types)
            message = self.generate_message(msg_type)
            message["sequence_id"] = i
            dataset.append(message)

        return dataset

    def save_dataset(
        self, dataset: List[Dict], filename: str = "agent_communications.json"
    ):
        """Save dataset to JSON file"""
        with open(filename, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset)} messages to {filename}")

    def get_message_statistics(self, dataset: List[Dict]) -> Dict:
        """Analyze message statistics"""
        total_messages = len(dataset)
        content_lengths = [msg["content_length"] for msg in dataset]
        message_types = [msg["message_type"] for msg in dataset]

        # Calculate JSON sizes
        json_sizes = []
        for msg in dataset:
            json_str = json.dumps(msg)
            json_sizes.append(len(json_str.encode("utf-8")))

        type_counts = {}
        for msg_type in message_types:
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1

        return {
            "total_messages": total_messages,
            "avg_content_length": np.mean(content_lengths),
            "avg_json_size": np.mean(json_sizes),
            "total_json_size": sum(json_sizes),
            "type_distribution": type_counts,
            "min_json_size": min(json_sizes),
            "max_json_size": max(json_sizes),
        }


if __name__ == "__main__":
    generator = RealAgentDataGenerator()

    # Generate dataset
    print("Generating realistic agent communication dataset...")
    dataset = generator.generate_dataset(2000)

    # Save dataset
    generator.save_dataset(dataset)

    # Show statistics
    stats = generator.get_message_statistics(dataset)
    print("\nDataset Statistics:")
    print(f"Total messages: {stats['total_messages']}")
    print(f"Average content length: {stats['avg_content_length']:.1f} chars")
    print(f"Average JSON size: {stats['avg_json_size']:.1f} bytes")
    print(f"Total dataset size: {stats['total_json_size']:,} bytes")
    print(f"Size range: {stats['min_json_size']}-{stats['max_json_size']} bytes")
    print("\nMessage type distribution:")
    for msg_type, count in stats["type_distribution"].items():
        percentage = (count / stats["total_messages"]) * 100
        print(f"  {msg_type}: {count} ({percentage:.1f}%)")
