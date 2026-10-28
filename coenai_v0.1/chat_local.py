from pathlib import Path
import ast
import operator
import random
import re

import chromadb
import ollama

WHITE = "\033[97m"
YELLOW = "\033[93m"
RESET = "\033[0m"

ROOT = Path(__file__).parent
DATABASE = ROOT / "data" / "chroma"
COLLECTION_NAME = "coen_history"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# Chroma's current collection uses its default L2 distance. This conservative
# cutoff is based on the reviewed notes in this project; re-calibrate if the
# embedding model or collection distance metric changes.
MAX_EVIDENCE_DISTANCE = 0.90

UNKNOWN_ANSWER = "I don't have a sufficiently relevant source in my local notes to answer that reliably, so I won't guess."
IDENTITY_ANSWER = (
    "I am an AI historical portrayal of Coen Dillen, based on source notes. "
    "I am not Coen himself and do not have his memories."
)
GREETINGS = {
    "en": [
        "Welcome to Eindhoven. I can talk about PSV, Dutch football in my playing years, or the history of Strijp and the city. What brings you here, and where would you like to begin?",
        "Good to have you here. We can explore PSV, football in the 1950s, or how Eindhoven has changed. What are you curious about?",
        "Welcome. My story is one part of Eindhoven's wider history. Would you like to begin with PSV, my football years, or the city?",
    ],
    "nl": [
        "Welkom in Eindhoven. Ik kan vertellen over PSV, het voetbal in mijn speelperiode of de geschiedenis van Strijp en de stad. Wat brengt u hier?",
        "Fijn dat u er bent. We kunnen PSV, het voetbal in de jaren vijftig of de veranderingen in Eindhoven bespreken. Waar bent u nieuwsgierig naar?",
        "Welkom. Mijn verhaal is één deel van de geschiedenis van Eindhoven. Wilt u beginnen bij PSV, mijn voetbaljaren of de stad?",
    ],
}


def extract_verified_claims(document):
    """Only use text explicitly marked as a verified fact in a source note."""
    sections = re.split(r"(?im)^\s*\\?##+\s*Verified fact\s*$", document)
    claims = []
    for section in sections[1:]:
        section = re.split(r"(?im)^\s*(?:Source|URL|Accessed):", section)[0]
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        lines = [line for line in lines if not re.match(r"^\\?#+\s", line)]
        if not lines:
            continue
        claim = " ".join(lines)
        claim = re.sub(r"^Claim\s*:\s*", "", claim, flags=re.IGNORECASE)
        claims.append(claim)
    return claims


def source_reference(document, filename):
    title = re.search(r"(?im)^\s*Source:\s*(.+)$", document)
    url = re.search(r"https?://\S+", document)
    label = title.group(1).strip() if title else filename
    parts = [label, filename]
    if url:
        parts.append(url.group(0).rstrip(".,)"))
    return " (Source: " + "; ".join(parts) + ")"


def extract_math(question):
    match = re.fullmatch(
        r"\s*(?:(?:what is|calculate|solve)\s+)?([0-9\s.+*/()%\-]+)\??\s*",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        tree = ast.parse(match.group(1), mode="eval")
    except SyntaxError:
        return None

    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.UAdd: operator.pos, ast.USub: operator.neg,
    }

    def calculate(node):
        if isinstance(node, ast.Expression):
            return calculate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](calculate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right = calculate(node.left), calculate(node.right)
            if abs(left) > 1_000_000 or abs(right) > 1_000_000:
                raise ValueError("number too large")
            return ops[type(node.op)](left, right)
        raise ValueError("unsupported arithmetic")

    try:
        result = calculate(tree)
        if not isinstance(result, (int, float)) or abs(result) > 1_000_000_000:
            return None
        return str(int(result)) if int(result) == result else str(result)
    except (ArithmeticError, ValueError, OverflowError):
        return None


def is_dutch(question):
    return bool(re.search(r"\b(wie|wat|hoeveel|wanneer|waar|waarom|vertel|over|bron|bronnen)\b", question, re.IGNORECASE))


def answer_for(question, collection, last_sources, last_answer_supported, last_greeting):
    normalized = question.casefold().strip().rstrip(".!?,")
    dutch = is_dutch(question)

    greeting_words = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
    dutch_greeting_words = {"hoi", "hallo", "goedemorgen", "goedemiddag", "goedenavond"}
    if normalized in greeting_words | dutch_greeting_words:
        options = GREETINGS["nl" if normalized in dutch_greeting_words else "en"]
        choices = [reply for reply in options if reply != last_greeting]
        return random.choice(choices or options), [], last_answer_supported, last_greeting

    if re.search(r"\b(who are you|are you (?:really )?coen|are you an ai|ben jij|wie ben jij)\b", normalized):
        return IDENTITY_ANSWER, [], False, last_greeting

    math_answer = extract_math(question)
    if math_answer is not None:
        prefix = "Dat is" if dutch else "That equals"
        return f"{prefix} {math_answer}.", [], False, last_greeting

    if re.search(r"\b(where did you get|what is your source|cite your source|source for that|waar komt deze informatie vandaan|welke bron)\b", normalized):
        if last_sources:
            return "My previous factual answer came from: " + "; ".join(last_sources), last_sources, True, last_greeting
        return "I don't have a source-backed factual answer to cite from my previous reply. Please treat it as unverified.", [], False, last_greeting

    if re.search(r"\b(you said|you told me|but you|earlier you|before you)\b", normalized):
        if not last_answer_supported:
            return "You're right to question that. My notes do not support the earlier claim, so please treat it as unverified.", [], False, last_greeting

    try:
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=question, keep_alive="10m").embeddings[0]
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(4, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return "I can't access my local notes right now, so I can't verify an answer.", [], False, last_greeting

    grounded_answers = []
    sources = []
    for document, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        if document is None or metadata is None or distance is None or distance > MAX_EVIDENCE_DISTANCE:
            continue
        filename = metadata.get("source", "source note")
        reference = source_reference(document, filename)
        for claim in extract_verified_claims(document):
            # First person applies only to a verified claim whose subject is Coen.
            if re.match(r"^Coen Dillen\b", claim, re.IGNORECASE):
                claim = re.sub(r"^Coen Dillen\s+is\b", "I am", claim, flags=re.IGNORECASE)
                claim = re.sub(r"^Coen Dillen\b", "I", claim, flags=re.IGNORECASE)
            grounded_answers.append(claim + reference)
            sources.append(reference.removeprefix(" (Source: ").removesuffix(")"))

    if not grounded_answers:
        fallback = (
            "Ik heb geen voldoende relevante bron in mijn lokale notities om dit betrouwbaar te beantwoorden, dus ik ga niet gokken."
            if dutch else UNKNOWN_ANSWER
        )
        return fallback, [], False, last_greeting

    # Return only the reviewed claim text and its recorded citation; don't ask
    # the chat model to embellish or invent connective historical details.
    unique_answers = list(dict.fromkeys(grounded_answers))
    return " ".join(unique_answers), list(dict.fromkeys(sources)), True, last_greeting


print(RESET, end="", flush=True)
client = chromadb.PersistentClient(path=str(DATABASE))
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    raise SystemExit("Database not found. Run index_sources.py first.")
if collection.count() == 0:
    raise SystemExit("The database is empty. Add source notes and run index_sources.py.")

print("Coen Dillen evidence-grounded local prototype. Type 'quit' to leave.")
last_greeting = None
last_sources = []
last_answer_supported = False

while True:
    question = input(f"\n{WHITE}You: ").strip()
    print(RESET, end="")
    if question.casefold() in {"quit", "exit"}:
        break
    if not question:
        continue
    answer, last_sources, last_answer_supported, last_greeting = answer_for(
        question, collection, last_sources, last_answer_supported, last_greeting
    )
    if answer in GREETINGS["en"] or answer in GREETINGS["nl"]:
        last_greeting = answer
    print(f"\n{YELLOW}Coen AI: {answer}{RESET}\n")

