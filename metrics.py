import numpy as np
import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

class AgentCommunicationMetrics:
    """Module to compute communication efficiency metrics for agent systems, 
    including original vs. compressed comparisons."""

    def __init__(self):
        self.metrics = {}

    def load_logs(self, logs: List[Dict]) -> pd.DataFrame:
        """Convert message logs to DataFrame for processing."""
        return pd.DataFrame(logs)

    # Speed-Related Metrics
    def acl(self, logs: List[Dict]) -> float:
        """Average Conversation Length: Average number of message exchanges per task/mission."""
        df = self.load_logs(logs)
        if 'mission_id' in df.columns:
            return len(df) / df['mission_id'].nunique()
        return len(df)  # Fallback if no mission_id

    def number_of_message_passing_rounds(self, logs: List[Dict]) -> int:
        """Number of Message-Passing Rounds: Total unique sequences or turns."""
        df = self.load_logs(logs)
        if 'sequence_id' in df.columns:
            return df['sequence_id'].nunique()
        return len(df)  # Assume each message is a round if no sequence

    def replanning_speed(self, logs: List[Dict]) -> float:
        """Replanning Speed: Average time between messages (proxy for latency)."""
        df = self.load_logs(logs)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            time_diffs = df['timestamp'].diff().dropna()
            return time_diffs.mean() if not time_diffs.empty else 0.0
        return 0.0

    def api_call_frequency(self, logs: List[Dict]) -> float:
        """API Call Frequency: Average calls per message (not directly applicable, stubbed)."""
        # In this system, no API calls; return message count as proxy
        return len(logs)

    def task_duration(self, logs: List[Dict]) -> float:
        """Task Duration: Total time from first to last message."""
        df = self.load_logs(logs)
        if 'timestamp' in df.columns:
            return df['timestamp'].max() - df['timestamp'].min()
        return 0.0

    # Memory Usage-Related Metrics
    def token_usage(self, logs: List[Dict]) -> float:
        """Token Usage (τ): Average byte size per message."""
        df = self.load_logs(logs)
        if 'content_length' in df.columns:
            return df['content_length'].mean()
        # Fallback to JSON size
        sizes = [len(json.dumps(msg)) for msg in logs]
        return np.mean(sizes)

    def communication_overhead(self, logs: List[Dict]) -> float:
        """Communication Overhead: Total byte size of all messages."""
        df = self.load_logs(logs)
        if 'content_length' in df.columns:
            return df['content_length'].sum()
        # Fallback to JSON size
        sizes = [len(json.dumps(msg)) for msg in logs]
        return np.sum(sizes)

    def computational_efficiency_memory(self, logs: List[Dict]) -> float:
        """Computational Efficiency (Memory): Average memory per message (proxy via size)."""
        return self.token_usage(logs)  # Use token usage as proxy

    # Loss of Detail-Related Metrics
    def partial_success_rate(self, original_logs: List[Dict], decompressed_logs: List[Dict]) -> float:
        """Partial Success Rate (PSR): Average progress/similarity score."""
        similarities = []
        for orig, decomp in zip(original_logs, decompressed_logs):
            # Simple match score (type + priority as in benchmark)
            score = 0
            if orig.get('message_type') == decomp.get('message_type'):
                score += 50
            if orig.get('priority') == decomp.get('priority'):
                score += 50
            similarities.append(score)
        return np.mean(similarities)

    def soft_evaluation_scores(self, original_logs: List[Dict], decompressed_logs: List[Dict]) -> float:
        """Soft Evaluation Scores: Average cosine similarity between embeddings."""
        similarities = []
        for orig, decomp in zip(original_logs, decompressed_logs):
            # Safely load embeddings and handle mismatched dimensions
            orig_emb = np.array(orig.get('embedding', np.zeros(384)), dtype=float)
            decomp_emb = np.array(decomp.get('decoded_embedding', np.zeros(384)), dtype=float)

            # If dimensions differ, pad the shorter vector with zeros to match
            if orig_emb.ndim != 1:
                orig_emb = orig_emb.ravel()
            if decomp_emb.ndim != 1:
                decomp_emb = decomp_emb.ravel()

            if orig_emb.size == 0 or decomp_emb.size == 0:
                continue

            if orig_emb.size != decomp_emb.size:
                max_len = max(orig_emb.size, decomp_emb.size)
                if orig_emb.size < max_len:
                    orig_emb = np.pad(orig_emb, (0, max_len - orig_emb.size))
                if decomp_emb.size < max_len:
                    decomp_emb = np.pad(decomp_emb, (0, max_len - decomp_emb.size))

            try:
                sim = float(cosine_similarity(orig_emb.reshape(1, -1), decomp_emb.reshape(1, -1))[0][0])
                similarities.append(sim)
            except Exception:
                # Skip problematic pairs
                continue
        return np.mean(similarities) if similarities else 0.0

    def damage_minimization(self, original_logs: List[Dict], decompressed_logs: List[Dict]) -> float:
        """Damage Minimization: Log-scaled penalty for mismatches (e.g., in critical types)."""
        penalties = []
        for orig, decomp in zip(original_logs, decompressed_logs):
            penalty = 0
            if orig.get('message_type') in ['emergency', 'obstacle'] and orig.get('message_type') != decomp.get('message_type'):
                penalty += 10  # High for safety-critical
            penalties.append(np.log(1 + penalty))
        return np.mean(penalties)

    # Precision-Related Metrics
    def average_mistakes(self, original_logs: List[Dict], decompressed_logs: List[Dict]) -> float:
        """Average Mistakes (AM): Average number of mismatches per message."""
        mistakes = []
        for orig, decomp in zip(original_logs, decompressed_logs):
            error_count = 0
            if orig.get('message_type') != decomp.get('message_type'):
                error_count += 1
            if orig.get('priority') != decomp.get('priority'):
                error_count += 1
            mistakes.append(error_count)
        return np.mean(mistakes)

    def error_types(self, original_logs: List[Dict], decompressed_logs: List[Dict]) -> Dict[str, int]:
        """Error Types (Qualitative): Count of mismatch types."""
        errors = {'type_mismatch': 0, 'priority_mismatch': 0, 'other': 0}
        for orig, decomp in zip(original_logs, decompressed_logs):
            if orig.get('message_type') != decomp.get('message_type'):
                errors['type_mismatch'] += 1
            if orig.get('priority') != decomp.get('priority'):
                errors['priority_mismatch'] += 1
        return errors

    def coordination_effectiveness(self, logs: List[Dict]) -> float:
        """Coordination Effectiveness: Proxy via type diversity (higher = better coordination)."""
        df = self.load_logs(logs)
        if 'message_type' in df.columns:
            unique_types = df['message_type'].nunique()
            total = len(df)
            return unique_types / total if total > 0 else 0.0
        return 0.0

    def observation_sharing_score(self, logs: List[Dict]) -> float:
        """Observation Sharing (OS): Fraction of 'obstacle' or 'status' messages."""
        df = self.load_logs(logs)
        if 'message_type' in df.columns:
            os_count = df[df['message_type'].isin(['obstacle', 'status'])].shape[0]
            return os_count / len(df) if len(df) > 0 else 0.0
        return 0.0

    def realtime_coordination_score(self, logs: List[Dict]) -> float:
        """Realtime Coordination (RC): Fraction of 'coordination' or 'navigation' messages."""
        df = self.load_logs(logs)
        if 'message_type' in df.columns:
            rc_count = df[df['message_type'].isin(['coordination', 'navigation'])].shape[0]
            return rc_count / len(df) if len(df) > 0 else 0.0
        return 0.0

    def calculate_all_metrics(self, original_logs: List[Dict], decompressed_logs: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Compute all applicable metrics. Provide decompressed_logs for comparison metrics."""
        self.metrics = {
            # Speed (on original)
            'acl': self.acl(original_logs),
            'number_of_message_passing_rounds': self.number_of_message_passing_rounds(original_logs),
            'replanning_speed': self.replanning_speed(original_logs),
            'api_call_frequency': self.api_call_frequency(original_logs),
            'task_duration': self.task_duration(original_logs),
            
            # Memory (on original)
            'token_usage': self.token_usage(original_logs),
            'communication_overhead': self.communication_overhead(original_logs),
            'computational_efficiency_memory': self.computational_efficiency_memory(original_logs),
            
            # Coordination (on original)
            'coordination_effectiveness': self.coordination_effectiveness(original_logs),
            'observation_sharing_score': self.observation_sharing_score(original_logs),
            'realtime_coordination_score': self.realtime_coordination_score(original_logs)
        }

        if decompressed_logs:
            self.metrics.update({
                # Loss
                'partial_success_rate': self.partial_success_rate(original_logs, decompressed_logs),
                'soft_evaluation_scores': self.soft_evaluation_scores(original_logs, decompressed_logs),
                'damage_minimization': self.damage_minimization(original_logs, decompressed_logs),
                
                # Precision
                'average_mistakes': self.average_mistakes(original_logs, decompressed_logs),
                'error_types': self.error_types(original_logs, decompressed_logs)
            })

        return self.metrics

# Example usage (can be integrated into demo)
if __name__ == "__main__":
    # Mock data for testing
    sample_original = [
        {'mission_id': 'mission_01', 'sequence_id': 1, 'timestamp': time.time(), 'message_type': 'navigation', 'priority': 'high', 'content': 'move to x=10', 'content_length': 12, 'embedding': np.random.rand(384)},
        {'mission_id': 'mission_01', 'sequence_id': 2, 'timestamp': time.time() + 1, 'message_type': 'status', 'priority': 'normal', 'content': 'battery 80%', 'content_length': 11, 'embedding': np.random.rand(384)}
    ]
    sample_decompressed = [
        {'message_type': 'navigation', 'priority': 'high', 'decoded_embedding': np.random.rand(384)},
        {'message_type': 'status', 'priority': 'low', 'decoded_embedding': np.random.rand(384)}  # Mismatch in priority
    ]
    
    metrics = AgentCommunicationMetrics()
    results = metrics.calculate_all_metrics(sample_original, sample_decompressed)
    print(results)