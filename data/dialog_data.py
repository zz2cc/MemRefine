"""Dataset loading and sample data generation.

Supports:
- Loading from HuggingFace datasets (MultiWOZ, SAMSum)
- Loading custom JSON/JSONL files
- Generating synthetic sample data for testing
"""

import json
import os
import random
from typing import List, Dict, Optional


class DialogDataset:
    """Container for dialogue data, providing iteration and statistics.

    Supports two formats:
      1. {"id": "...", "text": "..."} — dialogue text only
      2. {"id": "...", "dialogue": "...", "summary": "..."} — dialogue + reference memory
    """

    def __init__(self, dialogues: List[Dict[str, str]]):
        """dialogues: list of dicts with at least 'id' and either 'text' or 'dialogue'."""
        normalized = []
        for d in dialogues:
            entry = {"id": d.get("id", "unknown")}
            # Support both 'text' and 'dialogue' keys
            entry["text"] = d.get("text") or d.get("dialogue", "")
            # Preserve reference summary if available
            if "summary" in d:
                entry["reference_summary"] = d["summary"]
            normalized.append(entry)
        self.dialogues = normalized

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        return self.dialogues[idx]

    def __iter__(self):
        return iter(self.dialogues)

    def sample(self, n: int, seed: int = 42) -> "DialogDataset":
        """Return a random subset of n dialogues."""
        rng = random.Random(seed)
        indices = rng.sample(range(len(self)), min(n, len(self)))
        return DialogDataset([self.dialogues[i] for i in indices])

    def head(self, n: int = 5) -> "DialogDataset":
        return DialogDataset(self.dialogues[:n])

    @staticmethod
    def from_json(path: str) -> "DialogDataset":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect [{id, text}, ...]
        if isinstance(data, list):
            return DialogDataset(data)
        # Or {id: text, ...}
        if isinstance(data, dict):
            items = [{"id": k, "text": v} for k, v in data.items()]
            return DialogDataset(items)
        raise ValueError(f"Unsupported JSON format in {path}")

    @staticmethod
    def from_jsonl(path: str) -> "DialogDataset":
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return DialogDataset(items)

    @staticmethod
    def from_huggingface(dataset_name: str, split: str = "train", text_field: str = "dialogue",
                         id_field: Optional[str] = None, max_samples: int = None) -> "DialogDataset":
        """Load from HuggingFace datasets hub."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install `datasets` package: pip install datasets")

        ds = load_dataset(dataset_name, split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        items = []
        for i, row in enumerate(ds):
            text = row.get(text_field, str(row))
            dialog_id = row.get(id_field, f"{dataset_name}_{i}") if id_field else f"{dataset_name}_{i}"
            items.append({"id": str(dialog_id), "text": str(text)})

        return DialogDataset(items)


def create_sample_dataset(n: int = 50) -> DialogDataset:
    """Create a synthetic dialogue dataset for testing the pipeline.

    These are realistic customer service / multi-turn conversations covering
    various domains: tech support, order inquiries, booking, etc.
    """
    templates = [
        # --- Tech Support ---
        {
            "id": "tech_001",
            "text": (
                "Customer: Hi, my laptop keeps showing a blue screen error when I try to open "
                "Photoshop. It says 'MEMORY_MANAGEMENT' and restarts. I have a Dell XPS 15 with "
                "16GB RAM running Windows 11. Agent: I understand the frustration. Can you tell me "
                "when this started happening? Customer: About 3 days ago, right after I installed "
                "the latest Windows update (KB5034467). Before that everything was fine. Agent: "
                "Thank you. This could be a driver conflict caused by the update. Please try these "
                "steps: 1) Open Device Manager, find your display adapter, right-click and select "
                "'Roll Back Driver'. 2) If that doesn't work, boot into Safe Mode and uninstall "
                "the recent update from Settings > Windows Update > Update History. 3) Also run "
                "Windows Memory Diagnostic tool to rule out hardware issues. Can you try step 1 "
                "and let me know the result? Customer: Okay, I rolled back the NVIDIA driver as "
                "you suggested. The issue seems resolved now — Photoshop opens without the blue "
                "screen. Thank you! Agent: Great to hear! For future reference, you can pause "
                "automatic driver updates in Windows Update > Advanced Options. Is there anything "
                "else I can help with? Customer: No, that's all. Thanks again."
            ),
        },
        {
            "id": "tech_002",
            "text": (
                "User: I can't receive any emails on my Outlook since this morning. My internet "
                "is working fine, I can browse websites. Agent: Let me check your account. What's "
                "your email address? User: jane.smith@company.com. Agent: I see the issue — your "
                "mailbox has exceeded the storage limit of 50GB. You're currently at 49.8GB. "
                "This prevents new messages from being delivered. Would you like me to temporarily "
                "increase your quota while you clean up? User: Yes please, I didn't realize I was "
                "that close to the limit. Agent: Done — I've extended it to 75GB for 30 days. "
                "I'd recommend archiving old emails, especially those with large attachments. "
                "You can use the Mailbox Cleanup tool in Outlook > File > Tools. User: Got it, "
                "emails are coming through now. I'll start cleaning up today. Thanks!"
            ),
        },
        {
            "id": "tech_003",
            "text": (
                "Client: Our website (myshop.com) is down — customers are getting a 503 error. "
                "This is urgent, we're losing sales. Agent: I'm on it. Let me check the server "
                "status. The application server is running but the database connection pool is "
                "exhausted — all 200 connections are in use. This started at 14:30 UTC. Client: "
                "We did launch a flash sale at 2pm, traffic is about 5x normal. Agent: That "
                "explains it. I'm increasing the connection pool to 500 and restarting the "
                "application server. This will take about 2 minutes. Also, I'd recommend adding "
                "a read replica for the database and enabling connection pooling on the app side "
                "for the long term. Client: The site is back up! Let's schedule a call tomorrow "
                "to discuss the read replica. Agent: I've created a ticket for the follow-up and "
                "sent you my availability. The temporary pool increase will hold for now."
            ),
        },
        # --- Order / E-commerce ---
        {
            "id": "order_001",
            "text": (
                "Customer: I placed order #ORD-88421 three days ago and it still shows "
                "'Processing'. The estimated delivery was supposed to be today. Can you check "
                "what's going on? Agent: I apologize for the delay. Let me look into that. "
                "The order contains a Samsung 27-inch monitor and a wireless keyboard. Is that "
                "correct? Customer: Yes, that's right. Agent: The monitor is currently out of "
                "stock at our main warehouse, but we have inventory at the regional center in "
                "Dallas. I've re-routed your order. The new estimated delivery date is this "
                "Friday, May 19th. I've also added free express shipping as compensation. "
                "Customer: That works. Will I get a tracking number? Agent: Yes, you'll receive "
                "an email with the tracking number within 4 hours. I've also added a $15 store "
                "credit to your account for the inconvenience. Customer: I appreciate that. "
                "Thank you for handling it quickly."
            ),
        },
        {
            "id": "order_002",
            "text": (
                "Shopper: I received the wrong item in my package. I ordered a blue cotton "
                "t-shirt size M (SKU: TSH-BLU-M) but received a red polyester one size L "
                "(SKU: TSH-RED-L). Order number is WEB-55632. Agent: I'm sorry for the mix-up. "
                "I've verified the warehouse picking error from your photo. I'm initiating a "
                "replacement order now. The correct blue shirt will ship today via 2-day "
                "delivery. For the wrong item, please use the prepaid return label I'm emailing "
                "you — drop it at any UPS location within 14 days. Shopper: Do I need the "
                "original packaging? Agent: The original packaging would help, but any secure "
                "box is fine. Just include the return slip inside. You'll see the refund for "
                "any return shipping costs appear in 3-5 business days. Shopper: Perfect, "
                "thanks for making this easy."
            ),
        },
        # --- Booking / Travel ---
        {
            "id": "travel_001",
            "text": (
                "Caller: I need to change my flight booking. Confirmation number is FL-CN889. "
                "I'm scheduled on Delta flight 1278 from New York JFK to Los Angeles LAX on "
                "June 15th, but I need to leave on June 14th instead. Agent: Let me check "
                "availability. There's Delta flight 892 departing JFK at 8:30 AM on June 14th, "
                "arriving LAX at 11:45 AM. It has 12 seats available in economy. The change "
                "fee is $75 plus a $45 fare difference, total $120. Shall I proceed? Caller: "
                "That's fine. Please switch me to that flight. Agent: Done. Your new "
                "confirmation is FL-CN890, flight 892 on June 14th. I've emailed you the "
                "updated itinerary. Your seat is 22A, aisle. The $120 charge will appear on "
                "the card ending in 4402. Caller: Thank you, and what about my return flight? "
                "Agent: Your return flight 1279 LAX-JFK on June 22nd remains unchanged. "
                "Confirmation FL-CN699 is still valid. Caller: Great, that's all I needed."
            ),
        },
        {
            "id": "travel_002",
            "text": (
                "Guest: I'd like to book a room at your Chicago downtown hotel. Checking in "
                "November 3rd, checking out November 7th. Two adults, non-smoking. Agent: "
                "Let me check availability. We have a Deluxe King room with city view at $189 "
                "per night, or a Junior Suite at $249 per night. Both non-smoking. Which would "
                "you prefer? Guest: The Deluxe King sounds good. Does it include breakfast? "
                "Agent: Breakfast is available for an additional $18 per person per day. "
                "Alternatively, I can offer a package rate of $215 per night that includes "
                "breakfast for two and free parking, normally $30/day. Guest: Let's do the "
                "package. Total would be $860 for four nights? Agent: That's correct — $860 "
                "plus tax, approximately $987 total. I'll need a credit card to hold the "
                "reservation. Guest: Use the Visa ending in 7823. Agent: Reservation confirmed. "
                "Your confirmation number is CHI-4421. Check-in is 3pm, check-out 11am. "
                "Cancellation is free up to 48 hours before arrival."
            ),
        },
        # --- Medical / Healthcare ---
        {
            "id": "medical_001",
            "text": (
                "Patient: I need to schedule an annual physical exam. I'm a new patient — "
                "Dr. Chen was recommended to me. I have BlueCross BlueShield insurance, "
                "member ID BCS-99821. Agent: Let me check Dr. Chen's availability. She has "
                "openings next Tuesday at 10:30 AM or Thursday at 2:00 PM. Patient: Tuesday "
                "morning works. Agent: Before confirming, please bring your insurance card "
                "and a photo ID. As a new patient, arrive 20 minutes early to complete "
                "paperwork. You can also fill out the forms online through our patient portal "
                "to save time. The physical includes standard blood work — please fast for "
                "8 hours before the appointment. Patient: Noted. What's the copay? Agent: "
                "With your BCBS plan, the annual physical copay is $25. Any additional lab "
                "work beyond the standard panel may have separate charges. Your appointment "
                "is confirmed for Tuesday at 10:30 AM with Dr. Chen at our Main Street "
                "clinic, 450 Main Street, Suite 200. Patient: Thank you!"
            ),
        },
        # --- Finance / Banking ---
        {
            "id": "finance_001",
            "text": (
                "Client: I noticed an unauthorized charge on my checking account this morning. "
                "$347.50 from 'TechGadgets Online' — I've never shopped there. Account ending "
                "in 5632. Agent: I see the transaction, processed yesterday at 3:15 PM. I'm "
                "immediately blocking your current debit card and issuing a new one, which "
                "will arrive in 5-7 business days. For the disputed charge, I'm opening a "
                "Regulation E dispute. You'll receive provisional credit within 10 business "
                "days while we investigate. The case number is DISP-4421. Client: Can I still "
                "access my account in the meantime? Agent: Yes, you can use online banking, "
                "write checks, or visit a branch with your ID. I also see you have a savings "
                "account — I can transfer funds if you need immediate cash access. Client: "
                "Please transfer $500 from savings to checking. Agent: Done. Your checking "
                "balance is now $1,842.30. I've also enabled two-factor authentication on "
                "your account for added security. Would you like to review any other recent "
                "transactions? Client: No, that covers it. Thanks for the quick response."
            ),
        },
    ]

    # Generate more variants from templates
    dialogues = list(templates)
    seed = 42
    rng = random.Random(seed)

    # Pad with slight variations if n > len(templates)
    while len(dialogues) < n:
        base = rng.choice(templates)
        variant_text = base["text"].replace("Customer:", "Caller:", 1)  # trivial variant
        dialogues.append({
            "id": f"{base['id']}_v{len(dialogues)}",
            "text": variant_text,
        })

    return DialogDataset(dialogues[:n])


def load_custom_dataset(path: str, text_field: str = "text", id_field: str = "id") -> DialogDataset:
    """Load a custom dataset from a JSON/JSONL/CSV/TXT file.

    Supported formats:
    - .json: array of objects or dict mapping id→text
    - .jsonl: one JSON object per line
    - .txt: plain text, one dialogue per line
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        return DialogDataset.from_json(path)
    elif ext == ".jsonl":
        return DialogDataset.from_jsonl(path)
    elif ext == ".txt":
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    items.append({"id": f"line_{i}", "text": line})
        return DialogDataset(items)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
