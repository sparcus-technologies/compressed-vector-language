"""
Multi-Agent Demo - Practical Agent Communication Scenario

This demo shows a realistic multi-agent system where:
1. A Specialist Agent processes and understands a dataset
2. A Responder Agent answers user queries using the Specialist's knowledge
3. Knowledge is transferred efficiently via compressed KV caches

This is like having one agent be the "expert" on data, and another agent
use that expertise to answer questions - WITHOUT re-processing everything!
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from q_kvcomm import QKVCommConfig, QKVCommSystem


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🤖 Multi-Agent Communication Demo")
    print(f"Device: {device}\n")

    print("=" * 80)
    print("SCENARIO: Company Knowledge Base System")
    print("=" * 80)
    print("• Specialist Agent: Processes company documents")
    print("• Responder Agent: Answers employee questions")
    print("• Goal: Efficient knowledge sharing between agents\n")

    # Load models
    print("Loading agents...")

    # Specialist Agent - processes and understands documents
    specialist_model = "Qwen/Qwen2.5-1.5B-Instruct"
    specialist_tokenizer = AutoTokenizer.from_pretrained(specialist_model)
    specialist_tokenizer.pad_token = specialist_tokenizer.eos_token
    specialist = AutoModelForCausalLM.from_pretrained(specialist_model).to(device)
    specialist.tokenizer = specialist_tokenizer
    print(f"  ✓ Specialist Agent loaded ({specialist_model})")

    # Responder Agent - answers queries
    responder_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    responder_tokenizer = AutoTokenizer.from_pretrained(responder_model)
    responder_tokenizer.pad_token = responder_tokenizer.eos_token
    responder = AutoModelForCausalLM.from_pretrained(responder_model).to(device)
    responder.tokenizer = responder_tokenizer
    print(f"  ✓ Responder Agent loaded ({responder_model})")

    # Setup Q-KVComm for agent communication
    config = QKVCommConfig(
        mode="full",
        target_bits=6.0,
        quantization_enabled=True,
        calibration_enabled=True,
    )

    qkvcomm = QKVCommSystem(specialist, responder, config, device)

    # Calibration with sample company data
    print("\n📚 Calibrating with sample company knowledge...")
    calibration_data = [
        "Company policies are documented in the employee handbook.",
        "Our support team operates 24/7 with a guaranteed 2-hour response time.",
        "The annual review process begins in January each year.",
    ]
    qkvcomm.calibrate(calibration_data)
    print("✓ Calibration complete\n")

    # Company knowledge that Specialist Agent processes
    company_documents = {
        "hr_policy": (
            "CompanyX HR Policy: All employees are entitled to 20 days paid vacation per year. "
            "Vacation requests must be submitted at least 2 weeks in advance through the HR portal. "
            "Unused vacation days can be carried over to the next year, up to a maximum of 5 days. "
            "Sick leave is separate from vacation and requires a doctor's note for absences longer than 3 days."
        ),
        "benefits": (
            "CompanyX Employee Benefits: The company offers comprehensive health insurance covering "
            "medical, dental, and vision care. All full-time employees are enrolled automatically. "
            "We also provide a 401(k) retirement plan with 5% company matching, life insurance, "
            "and access to an employee assistance program. Gym membership reimbursement up to $50/month is available."
        ),
        "it_support": (
            "IT Support at CompanyX: For technical issues, employees can submit tickets through the IT portal "
            "or call extension 4357. Average response time is 30 minutes for critical issues, 4 hours for "
            "standard requests. Remote employees receive priority hardware replacement with next-day shipping. "
            "All company laptops come with full Microsoft Office suite and Adobe Creative Cloud licenses."
        ),
    }

    # Employee queries
    employee_queries = [
        "How many vacation days do I get per year?",
        "What health benefits does the company offer?",
        "How do I get IT support if I have a computer problem?",
        "Can I carry over my unused vacation days?",
    ]

    print("=" * 80)
    print("AGENT INTERACTIONS")
    print("=" * 80)

    for i, query in enumerate(employee_queries, 1):
        print(f"\n{'─' * 80}")
        print(f"Employee Query #{i}")
        print(f"{'─' * 80}")
        print(f"❓ Question: {query}")

        # Determine which document to use (in real system, could use retrieval)
        if "vacation" in query.lower():
            context = company_documents["hr_policy"]
            doc_name = "HR Policy"
        elif "health" in query.lower() or "benefit" in query.lower():
            context = company_documents["benefits"]
            doc_name = "Benefits Guide"
        elif "support" in query.lower() or "computer" in query.lower():
            context = company_documents["it_support"]
            doc_name = "IT Support Documentation"
        else:
            context = company_documents["hr_policy"]
            doc_name = "HR Policy"

        print(f"\n📄 Specialist Agent processing: {doc_name}")
        print(f"   Document: {context[:80]}...")

        print(f"\n🔄 Transferring knowledge: Specialist → Responder")
        print(f"   Using Q-KVComm compressed KV cache transfer...")

        # Specialist processes document, Responder answers using transferred knowledge
        output, metrics = qkvcomm.communicate(context, query, max_new_tokens=60)

        print(f"\n💬 Responder Agent's Answer:")
        print(f"   {output}")

        print(f"\n📊 Communication Metrics:")
        print(f"   • Compression: {metrics['avg_compression_ratio']:.2f}x")
        print(
            f"   • Bandwidth Saved: {(1 - 1/metrics['avg_compression_ratio'])*100:.1f}%"
        )
        print(f"   • Layers Transferred: {metrics['num_layers_transmitted']}")

    print("\n" + "=" * 80)
    print("✅ MULTI-AGENT DEMO COMPLETE")
    print("=" * 80)

    print("\n🎯 Key Benefits Demonstrated:")
    print("  • Specialist Agent processes documents once")
    print("  • Responder Agent reuses that understanding efficiently")
    print("  • No need to re-process documents for each query")
    print("  • Significant bandwidth savings in agent communication")
    print("  • Works across different model architectures!")

    print("\n💡 Real-World Applications:")
    print("  • Customer support systems with specialized knowledge agents")
    print("  • Document Q&A systems with reader and responder agents")
    print("  • Multi-agent research assistants with domain experts")
    print("  • Distributed AI systems with knowledge sharing")


if __name__ == "__main__":
    main()
