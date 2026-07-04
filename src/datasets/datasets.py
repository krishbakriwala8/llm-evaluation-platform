"""Real-world benchmark datasets for evaluation."""
import json
from pathlib import Path
from src.datasets import DatasetLoader, QAPair
from src.utils.config import DATA_DIR
from src.utils.logger import logger

# Finance QA Dataset
FINANCE_QA = [
    {
        "question": "What is the difference between stocks and bonds?",
        "answer": "Stocks represent ownership in a company, while bonds are debt instruments. Stocks offer higher potential returns but greater risk, while bonds provide fixed income with lower risk.",
        "context": "Stocks represent fractional ownership in a corporation. When you buy a stock, you become a shareholder and own a piece of the company. Bonds are fixed-income securities where an investor loans money to a borrower (corporate or government) for a set period and interest rate. Stocks are riskier but offer higher growth potential, while bonds provide regular interest payments and are generally considered safer.",
        "metadata": {"category": "investments", "difficulty": "beginner"},
    },
    {
        "question": "Explain compound interest and its importance in investing.",
        "answer": "Compound interest is earning interest on both the principal and previously earned interest. It's crucial for investing because it creates exponential growth over time, allowing wealth to multiply significantly.",
        "context": "Compound interest is the interest earned on both the original principal and accumulated interest from previous periods. Albert Einstein allegedly called it 'the eighth wonder of the world.' For example, if you invest $1000 at 5% annual compound interest, after 10 years you'll have $1629, not $1500. This accelerates wealth building over long periods. The formula is A = P(1 + r/n)^(nt). For long-term investors, compound interest is the most powerful wealth-building tool.",
        "metadata": {"category": "wealth_building", "difficulty": "intermediate"},
    },
    {
        "question": "What are the main components of a balance sheet?",
        "answer": "A balance sheet has three main components: assets (what the company owns), liabilities (what it owes), and shareholders' equity (owner's stake). The fundamental equation is: Assets = Liabilities + Equity.",
        "context": "The balance sheet is a financial statement showing a company's financial position at a specific date. Assets are resources the company owns and controls (cash, inventory, property). Liabilities are obligations or debts (loans, accounts payable). Shareholders' equity represents the owners' claim on assets after liabilities are paid. The balance sheet always balances: Total Assets = Total Liabilities + Shareholders' Equity. This is the accounting fundamental principle.",
        "metadata": {"category": "accounting", "difficulty": "intermediate"},
    },
    {
        "question": "What is diversification and why is it important?",
        "answer": "Diversification is spreading investments across different asset classes and sectors to reduce risk. It's important because it prevents losses in one investment from devastating the entire portfolio.",
        "context": "Diversification means not putting all eggs in one basket. By investing in various stocks, bonds, real estate, and commodities, an investor reduces idiosyncratic (company-specific) risk. While systematic market risk cannot be eliminated, diversification helps manage unsystematic risk. A well-diversified portfolio might include 60% stocks, 30% bonds, and 10% alternatives. The key principle is that different assets move differently under various market conditions, providing protection.",
        "metadata": {"category": "risk_management", "difficulty": "intermediate"},
    },
    {
        "question": "Explain what quantitative easing is.",
        "answer": "Quantitative easing (QE) is when central banks buy financial assets to increase money supply and lower interest rates, stimulating economic growth during recessions.",
        "context": "Quantitative easing is an unconventional monetary policy tool used when traditional interest rate cuts are ineffective (near zero percent). The central bank purchases large quantities of government bonds and other securities to inject money into the economy. This increases the money supply, lowers long-term interest rates, and encourages borrowing and spending. QE was extensively used after the 2008 financial crisis and during the COVID-19 pandemic.",
        "metadata": {"category": "monetary_policy", "difficulty": "advanced"},
    },
]

# Healthcare QA Dataset
HEALTHCARE_QA = [
    {
        "question": "What are the primary symptoms of type 2 diabetes?",
        "answer": "Common symptoms include increased thirst, frequent urination, fatigue, blurred vision, and slow wound healing. Many people may have no symptoms initially.",
        "context": "Type 2 diabetes develops when the body becomes resistant to insulin or doesn't produce enough. Typical symptoms include polydipsia (excessive thirst), polyuria (frequent urination), fatigue, blurred vision, slow healing of cuts or sores, and numbness in hands/feet. Some people are asymptomatic and discover diabetes during routine screening. Risk factors include obesity, family history, age over 45, and sedentary lifestyle.",
        "metadata": {"category": "endocrinology", "difficulty": "beginner"},
    },
    {
        "question": "How does the immune system fight infections?",
        "answer": "The immune system uses white blood cells, antibodies, and other mechanisms to identify and destroy pathogens. It involves both innate immunity (immediate response) and adaptive immunity (specific response).",
        "context": "The immune system has two main branches. The innate immune system provides immediate, non-specific defense using physical barriers, phagocytes, and inflammatory responses. The adaptive immune system provides specific, targeted responses through B cells (antibody production) and T cells (direct cell killing). When a pathogen enters, dendritic cells present antigens to T cells, triggering immune response. Memory cells remember the pathogen for faster future responses, which is the basis of vaccination.",
        "metadata": {"category": "immunology", "difficulty": "intermediate"},
    },
    {
        "question": "What is the role of the lymphatic system?",
        "answer": "The lymphatic system drains excess fluid from tissues, transports immune cells, absorbs dietary fats, and plays a crucial role in fighting infections.",
        "context": "The lymphatic system is a network of vessels and nodes working alongside the circulatory system. It collects lymph (fluid containing white blood cells) from tissues and returns it to the bloodstream. Lymph nodes filter lymph and trap pathogens and cancer cells. The spleen stores white blood cells and filters blood. The thymus produces T cells. This system is essential for immunity, preventing edema, and absorbing dietary fats from the intestines.",
        "metadata": {"category": "immunology", "difficulty": "intermediate"},
    },
    {
        "question": "Explain the concept of drug metabolism and bioavailability.",
        "answer": "Drug metabolism is how the body processes drugs to make them inactive, primarily in the liver. Bioavailability is the fraction of an administered dose that reaches systemic circulation.",
        "context": "Bioavailability (F) represents the percentage of an administered dose that reaches the systemic circulation in active form. Intravenous drugs have 100% bioavailability, while oral drugs may have lower bioavailability due to first-pass metabolism. Drug metabolism involves Phase I (oxidation/reduction), Phase II (conjugation), and Phase III (transport) reactions. The liver is the primary organ for drug metabolism through cytochrome P450 enzymes. Factors affecting metabolism include genetics, age, disease, and drug interactions.",
        "metadata": {"category": "pharmacology", "difficulty": "advanced"},
    },
    {
        "question": "What is the difference between DNA and RNA?",
        "answer": "DNA is double-stranded with deoxyribose sugar and thymine, while RNA is single-stranded with ribose sugar and uracil. DNA stores genetic information; RNA translates it into proteins.",
        "context": "DNA (deoxyribonucleic acid) and RNA (ribonucleic acid) are nucleic acids with structural differences. DNA has deoxyribose sugar, contains thymine, and is usually double-stranded. RNA has ribose sugar, contains uracil instead of thymine, and is typically single-stranded. DNA is the stable, long-term storage of genetic information, while RNA molecules (mRNA, tRNA, rRNA) are involved in protein synthesis. mRNA carries genetic code, tRNA brings amino acids, and rRNA is part of ribosomes.",
        "metadata": {"category": "molecular_biology", "difficulty": "intermediate"},
    },
]

# Technology QA Dataset
TECHNOLOGY_QA = [
    {
        "question": "What is the difference between machine learning and deep learning?",
        "answer": "Machine learning is a broad field enabling computers to learn from data. Deep learning is a subset using neural networks with multiple layers to learn hierarchical representations.",
        "context": "Machine Learning (ML) is a subset of artificial intelligence where algorithms learn patterns from data without explicit programming. Deep Learning (DL) is a specialized branch of ML using artificial neural networks with multiple layers (hence 'deep'). Deep learning excels with unstructured data like images, text, and audio by learning hierarchical features. For example, a DL model for image recognition learns edges in early layers, shapes in middle layers, and objects in later layers. While traditional ML works well with structured data, DL requires more data and computational resources but often achieves better performance on complex tasks.",
        "metadata": {"category": "artificial_intelligence", "difficulty": "intermediate"},
    },
    {
        "question": "Explain how blockchain technology works.",
        "answer": "Blockchain is a distributed ledger technology where transactions are grouped in blocks, cryptographically linked in a chain, and maintained by a decentralized network for transparency and security.",
        "context": "Blockchain is a distributed database maintaining a continuously growing list of records called blocks. Each block contains a cryptographic hash of the previous block, timestamp, and transaction data, creating an immutable chain. The blockchain is maintained by a peer-to-peer network rather than a central authority. Consensus mechanisms (Proof of Work, Proof of Stake) validate transactions. Once data is recorded, it's extremely difficult to alter retroactively because changing one block would require recalculating all subsequent blocks. This makes blockchain suitable for cryptocurrencies, supply chain tracking, and smart contracts.",
        "metadata": {"category": "cryptography", "difficulty": "intermediate"},
    },
    {
        "question": "What are microservices and why are they beneficial?",
        "answer": "Microservices are small, independent services performing specific business functions. They enable scalability, flexibility, and faster deployment compared to monolithic architectures.",
        "context": "Microservices architecture breaks applications into small, loosely coupled services, each handling a specific business capability. Each service runs in its own process, uses its own database, and communicates via APIs. Benefits include independent deployment, technology flexibility, easier scaling of specific services, improved fault isolation, and faster development. However, microservices introduce complexity in distributed systems, requiring robust monitoring, logging, and inter-service communication patterns. Popular patterns include API gateways, service discovery, circuit breakers, and message queues.",
        "metadata": {"category": "software_architecture", "difficulty": "advanced"},
    },
    {
        "question": "What is the purpose of containerization in DevOps?",
        "answer": "Containerization packages applications with dependencies in isolated environments, ensuring consistency across development, testing, and production, improving deployment reliability.",
        "context": "Containers use OS-level virtualization to package applications with their dependencies (libraries, runtime) in isolated, lightweight environments. Docker is the most popular containerization platform. Containers are smaller and faster than VMs, starting in milliseconds. They ensure 'it works on my machine' problems are eliminated because the container runs identically everywhere. Container orchestration platforms like Kubernetes manage deployment, scaling, and networking of containers. This is fundamental to modern DevOps and cloud-native development.",
        "metadata": {"category": "devops", "difficulty": "intermediate"},
    },
    {
        "question": "Explain the concept of API rate limiting.",
        "answer": "API rate limiting controls the number of requests clients can make in a time period, preventing abuse, ensuring fair usage, and protecting server resources.",
        "context": "Rate limiting restricts the number of API requests a client can make within a specified time window (e.g., 1000 requests per hour). It prevents abuse, DDoS attacks, ensures fair resource distribution, and maintains API stability. Common strategies include token bucket algorithm, sliding window, and leaky bucket. Rate limits should be communicated via HTTP headers (X-RateLimit-Limit, X-RateLimit-Remaining). Exceeding limits typically returns HTTP 429 (Too Many Requests). Authenticated users often have higher limits than anonymous users.",
        "metadata": {"category": "api_design", "difficulty": "intermediate"},
    },
]


def create_benchmark_datasets():
    """Create benchmark datasets in the data directory."""
    datasets_dir = DATA_DIR / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "finance_qa": FINANCE_QA,
        "healthcare_qa": HEALTHCARE_QA,
        "technology_qa": TECHNOLOGY_QA,
    }

    for name, data in datasets.items():
        path = datasets_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Created dataset: {name} with {len(data)} samples")


if __name__ == "__main__":
    create_benchmark_datasets()
