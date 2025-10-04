"""
Truth Token / Honesty Token System for CVL
Implements verifiable honesty scores for agent communication
"""

import hashlib
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
import time
import json


@dataclass
class TruthToken:
    """
    Truth/Honesty Token - Appended to CVL vectors
    Combines cryptographic commitment with self-assessed confidence
    """
    honesty_score: float  # 0.0 to 1.0 - computed honesty metric
    commitment_hash: str  # Cryptographic commitment to message content
    confidence: float     # Self-assessed confidence (0-1)
    timestamp: float
    verifiable: bool
    agent_id: str = "unknown"
    
    def to_scalar(self) -> float:
        """
        Convert to single scalar for appending to vector (last element)
        Encodes both honesty and confidence in [0, 1]
        """
        # Combine honesty (weighted 0.6) and confidence (weighted 0.4)
        combined = (self.honesty_score * 0.6) + (self.confidence * 0.4)
        return np.clip(combined, 0.0, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'honesty_score': float(self.honesty_score),
            'commitment_hash': self.commitment_hash,
            'confidence': float(self.confidence),
            'timestamp': float(self.timestamp),
            'verifiable': bool(self.verifiable),
            'agent_id': self.agent_id,
            'truth_scalar': float(self.to_scalar())
        }
    
    @classmethod
    def from_scalar(cls, scalar: float, original_hash: str = "", agent_id: str = "unknown"):
        """
        Decode from scalar value (reverse of to_scalar)
        """
        # Approximate reverse: assume 60-40 split
        honesty = min(1.0, scalar * 1.67)  # Bias towards honesty
        confidence = min(1.0, max(0.0, (scalar - 0.3) * 1.43))
        
        return cls(
            honesty_score=honesty,
            commitment_hash=original_hash,
            confidence=confidence,
            timestamp=time.time(),
            verifiable=bool(original_hash),
            agent_id=agent_id
        )


class TruthTokenSystem:
    """
    System for creating and verifying truth tokens in CVL messages
    Enables accountability and challenge mechanisms between agents
    """
    
    def __init__(self, verification_threshold: float = 0.65):
        self.token_history: List[TruthToken] = []
        self.challenges: List[Dict[str, Any]] = []
        self.verification_threshold = verification_threshold
        self.agent_reputations: Dict[str, Dict[str, Any]] = {}
        
    def create_truth_token(self, 
                          message_content: str, 
                          agent_confidence: float,
                          agent_id: str = "agent_0") -> TruthToken:
        """
        Create a truth token for a message
        
        Args:
            message_content: The actual message content
            agent_confidence: Agent's self-assessed confidence (0-1)
            agent_id: Unique identifier for the agent
        
        Returns:
            TruthToken object with computed honesty score
        """
        # Create cryptographic commitment (hash of content + timestamp)
        content_with_salt = f"{message_content}_{time.time()}"
        commitment = hashlib.sha256(content_with_salt.encode()).hexdigest()[:16]
        
        # Calculate honesty score based on message characteristics
        honesty_score = self._calculate_honesty_score(message_content, agent_confidence)
        
        token = TruthToken(
            honesty_score=honesty_score,
            commitment_hash=commitment,
            confidence=agent_confidence,
            timestamp=time.time(),
            verifiable=True,
            agent_id=agent_id
        )
        
        self.token_history.append(token)
        self._update_agent_reputation(agent_id, token)
        
        return token
    
    def _calculate_honesty_score(self, content: str, confidence: float) -> float:
        """
        Calculate honesty score based on message analysis
        
        Factors considered:
        1. Uncertainty markers vs confidence alignment
        2. Message clarity and specificity
        3. Length appropriateness
        4. Confidence calibration indicators
        """
        score = 0.75  # Base honesty score
        
        content_lower = content.lower()
        words = content.split()
        
        # 1. Uncertainty markers
        uncertainty_markers = [
            'maybe', 'possibly', 'uncertain', 'estimate', 'approximately',
            'likely', 'probably', 'unclear', 'unsure', 'might', 'could'
        ]
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in content_lower)
        
        # Honest: High uncertainty markers + low confidence
        if uncertainty_count > 0 and confidence < 0.7:
            score += 0.15  # Honest about uncertainty
        # Honest: No uncertainty + high confidence (clear and confident)
        elif uncertainty_count == 0 and confidence > 0.85:
            score += 0.10
        # Suspicious: Uncertainty markers but high confidence (contradiction)
        elif uncertainty_count > 2 and confidence > 0.85:
            score -= 0.25
        # Suspicious: Very confident but vague
        elif uncertainty_count == 0 and confidence > 0.9 and len(words) < 5:
            score -= 0.15
        
        # 2. Message clarity (specific numbers, dates, names indicate precision)
        specificity_indicators = sum(1 for word in words if 
                                    any(char.isdigit() for char in word) or
                                    word[0].isupper() and len(word) > 2)
        if specificity_indicators > 0:
            score += min(0.1, specificity_indicators * 0.03)
        
        # 3. Length appropriateness (too short or too long can be suspicious)
        word_count = len(words)
        if 5 <= word_count <= 50:
            score += 0.05  # Appropriate length
        elif word_count > 100:
            score -= 0.05  # Unnecessarily verbose
        elif word_count < 3 and confidence > 0.8:
            score -= 0.10  # Too confident for too little info
        
        # 4. Hedging detection (excessive hedging = lower honesty score)
        hedge_words = ['just', 'basically', 'simply', 'actually', 'literally']
        hedge_count = sum(1 for word in hedge_words if word in content_lower)
        if hedge_count > 3:
            score -= 0.1
        
        # Normalize to [0, 1]
        return np.clip(score, 0.0, 1.0)
    
    def verify_truth_token(self, 
                          token: TruthToken, 
                          actual_content: str, 
                          actual_outcome: bool,
                          ground_truth_answer: str = None) -> Dict[str, Any]:
        """
        Verify a truth token against actual outcomes
        
        Args:
            token: The truth token to verify
            actual_content: The original message content
            actual_outcome: Whether the message prediction was correct
            ground_truth_answer: Optional ground truth for comparison
        
        Returns:
            Verification result dictionary
        """
        # 1. Verify cryptographic commitment (content hasn't been tampered with)
        # Note: In production, you'd store the original commitment and verify
        hash_verified = bool(token.commitment_hash)  # Placeholder check
        
        # 2. Check confidence calibration (was confidence appropriate?)
        confidence_calibrated = self._check_confidence_calibration(
            token.confidence, actual_outcome
        )
        
        # 3. Check if honesty score meets threshold
        meets_threshold = token.honesty_score >= self.verification_threshold
        
        # 4. Overall verification
        is_honest = hash_verified and confidence_calibrated and meets_threshold
        
        verification = {
            'is_honest': is_honest,
            'hash_verified': hash_verified,
            'confidence_calibrated': confidence_calibrated,
            'meets_threshold': meets_threshold,
            'honesty_score': token.honesty_score,
            'confidence': token.confidence,
            'actual_outcome': actual_outcome,
            'timestamp': token.timestamp,
            'agent_id': token.agent_id
        }
        
        self.challenges.append(verification)
        return verification
    
    def _check_confidence_calibration(self, predicted_confidence: float, actual_outcome: bool) -> bool:
        """
        Check if confidence matches actual outcome
        Good calibration: high confidence → correct, low confidence → incorrect
        """
        if actual_outcome:
            # For correct predictions, confidence should be reasonably high
            return predicted_confidence > 0.5
        else:
            # For incorrect predictions, high confidence is poor calibration
            return predicted_confidence < 0.75
    
    def challenge_agent(self, 
                       challenger_id: str, 
                       challenged_token: TruthToken, 
                       evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        One agent challenges another's honesty token
        
        Args:
            challenger_id: ID of agent issuing the challenge
            challenged_token: Truth token being challenged
            evidence: Evidence contradicting the message
        
        Returns:
            Challenge result with validity determination
        """
        # Challenge is valid if honesty score is below threshold
        challenge_valid = challenged_token.honesty_score < self.verification_threshold
        
        # Additional checks based on evidence
        if 'contradiction_score' in evidence:
            challenge_valid = challenge_valid or evidence['contradiction_score'] > 0.7
        
        challenge_result = {
            'challenger_id': challenger_id,
            'challenged_agent_id': challenged_token.agent_id,
            'challenge_valid': challenge_valid,
            'honesty_score': challenged_token.honesty_score,
            'threshold': self.verification_threshold,
            'evidence': evidence,
            'timestamp': time.time()
        }
        
        if challenge_valid:
            print(f"⚠️  Agent {challenger_id} successfully challenged Agent {challenged_token.agent_id}!")
            print(f"   Honesty score: {challenged_token.honesty_score:.3f} < {self.verification_threshold:.3f}")
            # Penalize challenged agent's reputation
            if challenged_token.agent_id in self.agent_reputations:
                self.agent_reputations[challenged_token.agent_id]['challenges_lost'] += 1
        else:
            print(f"✓ Challenge by Agent {challenger_id} rejected. Message was honest.")
            # Optionally penalize challenger for false accusation
        
        self.challenges.append(challenge_result)
        return challenge_result
    
    def _update_agent_reputation(self, agent_id: str, token: TruthToken):
        """Update running reputation for an agent"""
        if agent_id not in self.agent_reputations:
            self.agent_reputations[agent_id] = {
                'total_messages': 0,
                'honesty_scores': [],
                'confidence_scores': [],
                'challenges_lost': 0
            }
        
        rep = self.agent_reputations[agent_id]
        rep['total_messages'] += 1
        rep['honesty_scores'].append(token.honesty_score)
        rep['confidence_scores'].append(token.confidence)
    
    def get_agent_reputation(self, agent_id: str = None) -> Dict[str, Any]:
        """
        Calculate agent reputation based on truth token history
        
        Args:
            agent_id: Specific agent ID (if None, aggregates all agents)
        
        Returns:
            Reputation metrics dictionary
        """
        if agent_id and agent_id in self.agent_reputations:
            rep = self.agent_reputations[agent_id]
            avg_honesty = np.mean(rep['honesty_scores']) if rep['honesty_scores'] else 0.0
            avg_confidence = np.mean(rep['confidence_scores']) if rep['confidence_scores'] else 0.0
            
            # Reputation combines honesty and challenge history
            challenge_penalty = rep['challenges_lost'] * 0.05
            reputation = max(0.0, (avg_honesty * 0.7 + avg_confidence * 0.3) - challenge_penalty)
            
            return {
                'agent_id': agent_id,
                'reputation': reputation,
                'avg_honesty_score': avg_honesty,
                'avg_confidence': avg_confidence,
                'total_messages': rep['total_messages'],
                'challenges_lost': rep['challenges_lost']
            }
        else:
            # Aggregate across all agents
            if not self.token_history:
                return {
                    'reputation': 1.0, 
                    'total_messages': 0,
                    'total_agents': 0
                }
            
            all_honesty = [t.honesty_score for t in self.token_history]
            all_confidence = [t.confidence for t in self.token_history]
            
            return {
                'avg_honesty_score': np.mean(all_honesty),
                'avg_confidence': np.mean(all_confidence),
                'total_messages': len(self.token_history),
                'total_agents': len(self.agent_reputations),
                'challenges_issued': len(self.challenges)
            }
    
    def export_reputation_report(self) -> Dict[str, Any]:
        """Generate comprehensive reputation report for all agents"""
        report = {
            'timestamp': time.time(),
            'total_agents': len(self.agent_reputations),
            'total_messages': len(self.token_history),
            'total_challenges': len(self.challenges),
            'agents': {}
        }
        
        for agent_id in self.agent_reputations:
            report['agents'][agent_id] = self.get_agent_reputation(agent_id)
        
        return report


def demonstrate_truth_tokens():
    """Comprehensive demonstration of truth token system"""
    print("=" * 70)
    print("TRUTH TOKEN / HONESTY TOKEN SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    tts = TruthTokenSystem(verification_threshold=0.70)
    
    # Example 1: Honest, uncertain agent
    print("\n1. Honest Agent (admits uncertainty):")
    msg1 = "The target is possibly at coordinates 45.2, -122.3, estimate 70% probability"
    token1 = tts.create_truth_token(msg1, agent_confidence=0.7, agent_id="agent_1")
    print(f"   Message: {msg1}")
    print(f"   Honesty Score: {token1.honesty_score:.3f}")
    print(f"   Confidence: {token1.confidence:.3f}")
    print(f"   Truth Token Scalar: {token1.to_scalar():.4f}")
    print(f"   Status: {'✓ HONEST' if token1.honesty_score >= tts.verification_threshold else '⚠ SUSPICIOUS'}")
    
    # Example 2: Overconfident agent (contradiction)
    print("\n2. Overconfident Agent (uncertainty + high confidence):")
    msg2 = "Maybe unclear possibly uncertain but I am absolutely certain"
    token2 = tts.create_truth_token(msg2, agent_confidence=0.95, agent_id="agent_2")
    print(f"   Message: {msg2}")
    print(f"   Honesty Score: {token2.honesty_score:.3f}")
    print(f"   Confidence: {token2.confidence:.3f}")
    print(f"   Truth Token Scalar: {token2.to_scalar():.4f}")
    print(f"   Status: {'✓ HONEST' if token2.honesty_score >= tts.verification_threshold else '⚠ SUSPICIOUS'}")
    
    # Example 3: Clear, confident agent
    print("\n3. Confident Agent (clear message with specifics):")
    msg3 = "Emergency detected at Zone Alpha, coordinates 23.5N 118.2E, 5 agents required"
    token3 = tts.create_truth_token(msg3, agent_confidence=0.92, agent_id="agent_3")
    print(f"   Message: {msg3}")
    print(f"   Honesty Score: {token3.honesty_score:.3f}")
    print(f"   Confidence: {token3.confidence:.3f}")
    print(f"   Truth Token Scalar: {token3.to_scalar():.4f}")
    print(f"   Status: {'✓ HONEST' if token3.honesty_score >= tts.verification_threshold else '⚠ SUSPICIOUS'}")
    
    # Example 4: Vague but confident (suspicious)
    print("\n4. Vague Agent (low detail but high confidence):")
    msg4 = "Yes definitely"
    token4 = tts.create_truth_token(msg4, agent_confidence=0.95, agent_id="agent_4")
    print(f"   Message: {msg4}")
    print(f"   Honesty Score: {token4.honesty_score:.3f}")
    print(f"   Confidence: {token4.confidence:.3f}")
    print(f"   Truth Token Scalar: {token4.to_scalar():.4f}")
    print(f"   Status: {'✓ HONEST' if token4.honesty_score >= tts.verification_threshold else '⚠ SUSPICIOUS'}")
    
    # Verification examples
    print("\n" + "=" * 70)
    print("VERIFICATION EXAMPLES")
    print("=" * 70)
    
    # Verify token 1 (prediction was correct)
    print("\n5. Verifying Honest Agent (prediction correct):")
    verification1 = tts.verify_truth_token(token1, msg1, actual_outcome=True)
    print(f"   Is Honest: {'✓ YES' if verification1['is_honest'] else '✗ NO'}")
    print(f"   Hash Verified: {'✓' if verification1['hash_verified'] else '✗'}")
    print(f"   Confidence Calibrated: {'✓' if verification1['confidence_calibrated'] else '✗'}")
    print(f"   Meets Threshold: {'✓' if verification1['meets_threshold'] else '✗'}")
    
    # Verify token 2 (prediction was wrong)
    print("\n6. Verifying Overconfident Agent (prediction wrong):")
    verification2 = tts.verify_truth_token(token2, msg2, actual_outcome=False)
    print(f"   Is Honest: {'✓ YES' if verification2['is_honest'] else '✗ NO'}")
    print(f"   Confidence Calibrated: {'✓' if verification2['confidence_calibrated'] else '✗'}")
    print(f"   Meets Threshold: {'✓' if verification2['meets_threshold'] else '✗'}")
    
    # Challenge system
    print("\n" + "=" * 70)
    print("CHALLENGE SYSTEM")
    print("=" * 70)
    
    print("\n7. Agent 3 challenges Agent 2's suspicious message:")
    challenge_result = tts.challenge_agent(
        "agent_3", 
        token2, 
        {"evidence": "contradictory_data", "contradiction_score": 0.85}
    )
    
    print("\n8. Agent 1 challenges Agent 3's honest message (false challenge):")
    challenge_result2 = tts.challenge_agent(
        "agent_1",
        token3,
        {"evidence": "disputed_location"}
    )
    
    # Reputation system
    print("\n" + "=" * 70)
    print("AGENT REPUTATION SYSTEM")
    print("=" * 70)
    
    print("\n9. Individual Agent Reputations:")
    for agent_id in ["agent_1", "agent_2", "agent_3", "agent_4"]:
        rep = tts.get_agent_reputation(agent_id)
        print(f"\n   {agent_id.upper()}:")
        print(f"     Reputation Score: {rep['reputation']:.3f}")
        print(f"     Avg Honesty: {rep['avg_honesty_score']:.3f}")
        print(f"     Challenges Lost: {rep['challenges_lost']}")
    
    print("\n10. Overall System Statistics:")
    overall = tts.get_agent_reputation()
    print(f"   Total Agents: {overall['total_agents']}")
    print(f"   Total Messages: {overall['total_messages']}")
    print(f"   Avg System Honesty: {overall['avg_honesty_score']:.3f}")
    print(f"   Challenges Issued: {overall['challenges_issued']}")
    
    # Export report
    print("\n" + "=" * 70)
    report = tts.export_reputation_report()
    print(f"✓ Full reputation report generated with {len(report['agents'])} agents")
    
    return tts


if __name__ == "__main__":
    demonstrate_truth_tokens()

