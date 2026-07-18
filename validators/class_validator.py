"""
validators/class_validator.py
──────────────────────────────
Class Diagram rule-based validation.

UML RELATIONSHIP CLASSIFIER  (sentence-level trigger rules)
  Applied BEFORE all structural/semantic checks.  Scans the raw scenario
  text sentence-by-sentence and classifies each X→Y pair using four rule sets:

  Association (General)
    Logic  : Independent interaction; both entities can exist without the other.
    Triggers: "X has Y" | "X uses Y" | "X is related to Y"
              "X is associated with Y"

  Aggregation (Weak Part-Of)
    Logic  : Part survives if the whole is deleted.
    Triggers: "X contains Y" | "X is a collection of Y" | "Y is a member of X"
              "X manages Y"  | "X holds Y"              | "X is made up of Y"

  Composition (Strong Part-Of)
    Logic  : Part is destroyed if the whole is deleted.
    Triggers: "X is composed of Y"     | "Y cannot exist without X"
              "X strictly owns Y"      | "X owns Y"
              "X consists of Y"

  Generalization / Specialization (Inheritance)
    Logic  : IS-A hierarchy — child inherits from parent.
    Triggers: "X is a type of Y"   | "X is a kind of Y"
              "X inherits from Y"  | "X extends Y"
              "X is a Y" / "X is an Y"
              "A and B inherit from C"  (multi-child form)

  Classified pairs are merged into scenario_rels so ALL downstream rules
  (R_REL, R_MISSING_REL, S3, R_GLOBAL_WRONG_TYPE) automatically benefit.

STRUCTURAL RULES:
  R1.  Missing classes (scenario -> diagram) — flexible matching.
       NOTE: 'member'/'section' are valid domain class names, not noise words.
  R4.  Relationship endpoint validation (only if user drew relationships).
  R5.  Extra classes (diagram -> not in scenario) — warning to remove.
  R6.  Unverified relationships — only if scenario defines relationships.
  R7.  Duplicate class names.
  R8.  Empty / missing class name — labeled "Class 1", "Class 2" etc.
  R11. Attribute spelling check.
  R12. Missing visibility prefix on attributes/methods.
  R13. Missing type annotation on attributes.
  R14. Method missing parentheses.
  R15. Invalid / missing multiplicity.
       RULE: Multiplicity is ALWAYS required on both ends of any explicit
       association / aggregation / composition arrow drawn by the user.
       Generic/unlabeled lines are only checked when the pair is in scenario_rels.
  R16. Isolated class — only if scenario implies relationships exist.
  R_REL.        Wrong relationship type drawn vs scenario expected.
  R_MISSING_REL. Relationship in scenario not drawn at all.
  R_ASSOC_LABEL. Association label check:
       (a) Scenario names a label for a pair → user must draw that label (ERROR).
           Label is detected from: explicit label field, classifier trigger phrase,
           OR (fallback) the raw scenario text scanned sentence-by-sentence for a
           known label verb (e.g. "manages") that appears in the same sentence as
           both class names.  This ensures NLP-only entries (which lack a "trigger"
           field) still trigger the label error when the scenario verb is present.
       (b) Scenario does not name a label   → no error either way (labels optional).

SEMANTIC RULES:
  S1.  Inheritance direction — child arrow must point TO parent.
  S2.  Self-association — class connected to itself.
  S3.  IS-A vs HAS-A mismatch — wrong relationship category.
  S4.  Semantically invalid multiplicity — e.g. 0..0.
  S5.  Multiplicity on generalization — never valid in UML.
  S6.  Redundant generalization — A->C when A->B->C already exists.
  S7.  Circular inheritance — A inherits B AND B inherits A.
  S8.  Abstract class heuristic — 3+ children but no attributes defined.

REMOVED:
  - R10 PascalCase enforcement.
  - R2/R3 missing attributes/methods (not validated in this version).
  - Isolated class / relationship errors suppressed when scenario has no relationships.
"""
import re
import logging
import difflib
from typing import List, Dict, Any, Set, Optional
from .base_validator import BaseValidator, ValidationError

_log = logging.getLogger(__name__)
# ── Fuzzy spelling checker ────────────────────────────────────────────────────
# Instead of a static dictionary, we compare each attribute/method token
# against words extracted from the scenario using similarity ratio.
# If a token is close (but not identical) to a scenario word → spelling error.
def _extract_scenario_words(extracted: Dict[str, Any]) -> Set[str]:
    """Pull all meaningful words from the scenario extracted data."""
    words: Set[str] = set()
    for v in extracted.values():
        if isinstance(v, str):
            for w in re.findall(r"[a-zA-Z]{3,}", v):
                words.add(w.lower())
        elif isinstance(v, list):
            for item in v:
                for w in re.findall(r"[a-zA-Z]{3,}", str(item)):
                    words.add(w.lower())
    return words
def _find_spelling_mistake(token: str, scenario_words: Set[str], cutoff: float = 0.75) -> Optional[str]:
    """
    Returns the closest scenario word if token looks like a misspelling of it.
    - token must NOT exactly match the scenario word (that's correct spelling)
    - similarity must be >= cutoff  (0.75 = close but not identical)
    - token must differ by at least 1 character from the match
    Returns the suggested correct word, or None if no mistake found.
    """
    token_l = token.lower()
    if token_l in scenario_words:
        return None  # exact match → correct spelling, no error
    matches = difflib.get_close_matches(token_l, scenario_words, n=1, cutoff=cutoff)
    if matches:
        best = matches[0]
        # Extra guard: avoid flagging very short or identical-length-but-different tokens
        if best != token_l:
            return best
    return None
# ── Valid multiplicity patterns ───────────────────────────────────────────────
VALID_MULTIPLICITY_RE = re.compile(
    r"^(\*|0|1|[0-9]+)(\.\.([\*]|[0-9]+))?$"
)
# ── Words / tokens that are NEVER valid class names ─────────────────────────
#
# CHANGE LOG
# ----------
# REMOVED — valid domain entities that were incorrectly blocked:
#   • entity / entities      → common OOP base class name
#   • component / components → valid in component-based system diagrams
#   • module / modules       → valid in module-level designs
#   • element / elements     → valid domain class (e.g. FormElement)
#   • node / nodes           → valid in tree / network diagrams
#   • event / events         → core domain class in event-driven systems
#   • state / states         → core domain class in state-machine diagrams
#   • action / actions       → valid domain class (e.g. UserAction)
#   • interface / interfaces → valid structural class in OOP designs
#   • record / records       → valid domain class (e.g. MedicalRecord)
#   • entry / entries        → valid domain class (e.g. JournalEntry)
#   • message / messages     → core domain class in messaging systems
#   • form / forms           → valid domain class in form-handling systems
#   • policy / policies      → valid domain class in rule/policy systems
#   • reference / references → in _DOMAIN_CLASS_WHITELIST; removed from noise
#   • connection / connections → in _DOMAIN_CLASS_WHITELIST; removed from noise
#   • collection             → in _DOMAIN_CLASS_WHITELIST; removed from noise
#   • condition / conditions → in _DOMAIN_CLASS_WHITELIST; removed from noise
#   • operation / operations → in _DOMAIN_CLASS_WHITELIST; removed from noise
#   • property / properties  → in _DOMAIN_CLASS_WHITELIST; removed from noise
#   • session                → in _DOMAIN_CLASS_WHITELIST; removed from noise
#   • assignment/assignments → in _DOMAIN_CLASS_WHITELIST; removed from noise
#   • supplier               → valid domain actor class
#   • client                 → valid domain actor class
#   • container              → valid domain class in container/packaging systems
#   • base                   → valid as part of compound class names
#
# ADDED — missing diagram-instruction / filler words:
#   • sketch / sketching / sketched
#   • filler / fillers
#   • placeholder / placeholders
#   • caption / captions
#   • annotate / annotating / annotated / annotation / annotations
#   • label / labels  (moved here from generic nouns — it's a diagramming term)
#   • tag / tags      (moved here — diagramming / metadata term)
#   • note / notes    (moved here — diagramming annotation term)
#   • outline / outlines
#   • draft / drafting / drafted
#   • revise / revising / revised / revision / revisions
#   • update / updating / updated
#   • modify / modifying / modified / modification / modifications
#
# DOMAIN_CLASS_WHITELIST CONFLICT RESOLUTION:
#   All words that appear in _DOMAIN_CLASS_WHITELIST have been removed from
#   _NOISE_WORDS so the suffix-check + whitelist path in _is_instruction_word()
#   can fire correctly.  Previously those words were short-circuited at the
#   exact-match noise check (step 1) and never reached the whitelist (step 9).
# ─────────────────────────────────────────────────────────────────────────────
_NOISE_WORDS: Set[str] = {
    # ── Articles / prepositions / conjunctions ────────────────────────────────
    "the", "a", "an", "and", "or", "of", "for", "in", "on", "at", "to",
    "by", "as", "is", "it", "its", "be", "are", "was", "were", "has",
    "have", "had", "do", "does", "did", "not", "no", "nor", "so", "yet",
    "both", "either", "neither", "each", "every", "any", "some", "all",
    "with", "within", "without", "through", "via", "using", "between",
    "among", "across", "above", "below", "under", "over", "into", "onto",
    "from", "about", "after", "before", "since", "until", "while", "if",
    "then", "than", "that", "this", "these", "those", "which", "who",
    "whom", "whose", "where", "when", "how", "why", "what",

    # ── Scenario / instruction / task words ──────────────────────────────────
    "draw", "drawing", "drawn",
    "create", "creating", "created",
    "make", "making", "made",
    "design", "designing", "designed",
    "build", "building", "built",
    "generate", "generating",
    "construct", "constructing",
    "develop", "developing",
    "produce", "producing",
    "write", "writing",
    "describe", "describing",
    "represent", "representing",
    "model", "modelling", "modeling",
    "define", "defining",
    "illustrate", "illustrating",
    "show", "showing", "shown",
    "display", "displaying",
    "add", "adding",
    "include", "including",
    "consider", "considering",
    "use", "using", "used",
    "given", "following",
    "need", "needs", "needed",
    "want", "wants",
    "must", "should", "shall", "can", "could", "would", "will", "may", "might",
    # newly added diagram-instruction / filler words
    "sketch", "sketching", "sketched",
    "draft", "drafting", "drafted",
    "outline", "outlines",
    "revise", "revising", "revised",
    "update", "updating", "updated",
    "modify", "modifying", "modified",
    "filler", "fillers",
    "placeholder", "placeholders",
    "annotate", "annotating", "annotated",

    # ── Diagram / notation meta-words ─────────────────────────────────────────
    "diagram", "diagrams",
    "uml", "umldiagram",
    "notation", "notations",
    "scenario", "scenarios",
    "classification", "classifications",
    "hierarchy", "hierarchies",
    "structure", "structures",
    "pattern", "patterns",
    "chart", "charts",
    "scheme", "schemes",
    "blueprint", "blueprints",
    "overview",
    "representation", "representations",
    "example", "examples",
    "template", "templates",
    "sample", "samples",
    "exercise", "exercises",
    # "assignment" / "assignments" REMOVED — in _DOMAIN_CLASS_WHITELIST
    "task", "tasks",
    "problem", "problems",
    "question", "questions",
    "answer", "answers",
    "solution", "solutions",
    "following", "follows",
    "complete", "simple", "basic", "detailed", "full", "partial",
    "correct", "incorrect", "proper", "improper",
    "appropriate", "inappropriate",
    "valid", "invalid",
    "below", "above", "here", "there",
    "now", "then", "already", "also", "only", "just", "even", "still",
    "well", "very", "quite", "rather", "too", "much", "many", "more",
    "most", "least", "less", "few", "several", "other", "another",
    "same", "different", "similar", "new", "old",
    "first", "second", "third", "last", "next", "previous",
    "one", "two", "three", "four", "five",
    "multiple", "various", "certain", "specific",
    # annotation / revision meta-words (newly added)
    "annotation", "annotations",
    "revision", "revisions",
    "modification", "modifications",
    "caption", "captions",

    # ── Class / OOP / UML meta-words ──────────────────────────────────────────
    # NOTE: entity/entities, component/components, module/modules,
    #       element/elements, node/nodes, event/events, state/states,
    #       action/actions, interface/interfaces REMOVED — valid domain classes.
    "class", "classes",
    "object", "objects",
    "instance", "instances",
    "attribute", "attributes",
    "method", "methods",
    "function", "functions",
    "procedure", "procedures",
    "behavior", "behaviour", "behaviors", "behaviours",
    "field", "fields",
    "variable", "variables",
    "parameter", "parameters",
    "argument", "arguments",
    "return", "returns",
    "type", "types",
    "value", "values",
    "data", "datum",
    "abstract", "abstraction",
    "concrete",
    "override", "overrides", "overriding",
    "implement", "implements", "implementing", "implementation",
    "extend", "extends", "extending", "extension",
    "inherit", "inherits", "inheriting", "inheritance",
    "overload",
    "polymorphism", "encapsulation",
    "visibility", "access", "scope",
    "public", "private", "protected", "package",
    "static", "final", "virtual", "readonly",
    "constructor", "destructor",
    "getter", "setter",

    # ── Relationship meta-words ────────────────────────────────────────────────
    # NOTE: connection/connections, reference/references REMOVED — in whitelist.
    #       supplier, client, container, base REMOVED — valid domain actors/classes.
    "relationship", "relationships",
    "relation", "relations",
    "generalization", "generalisation",
    "generalizations", "generalisations",
    "specialization", "specialisation",
    "association", "associations",
    "aggregation", "aggregations",
    "composition", "compositions",
    "dependency", "dependencies",
    "realization", "realisation",
    "realizations", "realisations",
    "connected", "connecting",
    "link", "links", "linked", "linking",
    "arrow", "arrows",
    "line", "lines",
    "edge", "edges",
    "path", "paths",
    "pointer", "pointers",
    "direction", "directions",
    "multiplicity", "multiplicities",
    "cardinality", "cardinalities",
    "isa", "hasa",
    "parent", "child", "children",
    "superclass", "subclass", "superclasses", "subclasses",
    "derived",
    "owner", "owned",
    "whole", "part", "parts",
    "contained",
    "having", "contains", "contain",

    # ── Primitive / built-in types ────────────────────────────────────────────
    # NOTE: "collection" REMOVED — in _DOMAIN_CLASS_WHITELIST.
    "string", "str",
    "int", "integer",
    "float", "double", "decimal",
    "boolean", "bool",
    "char", "character",
    "byte", "short", "long",
    "void", "null", "none", "undefined",
    "list", "array", "vector",
    "map", "dictionary", "dict",
    "set",
    "queue", "stack",
    "tuple", "pair",
    "date", "datetime", "time", "timestamp",
    "number", "num",

    # ── Generic nouns that are NOT typically class names ──────────────────────
    # NOTE: record/records, entry/entries, message/messages, form/forms,
    #       policy/policies, condition/conditions, operation/operations,
    #       property/properties, event/events, state/states, action/actions,
    #       interface/interfaces, session REMOVED — valid domain class names.
    # NOTE: label/labels, tag/tags, note/notes moved to diagram-meta section.
    "name", "names",
    "id", "ids",
    "identifier", "identifiers",
    "amount", "amounts",
    "balance", "balances",
    "info", "information",
    "detail", "details",
    "item", "items",
    "result", "results",
    "output", "outputs",
    "input", "inputs",
    "point", "points",
    "level", "levels",
    "step", "steps",
    "stage", "stages",
    "phase", "phases",
    "mode", "modes",
    "status", "statuses",
    "flag", "flags",
    "code", "codes",
    "key", "keys",
    "label", "labels",
    "tag", "tags",
    "note", "notes",
    "comment", "comments",
    "description", "descriptions",
    "text", "texts",
    "content", "contents",
    "body", "bodies",
    "header", "headers",
    "title", "titles",
    "category", "categories",
    "group", "groups",
    "kind", "kinds",
    "format", "formats",
    "style", "styles",
    "rule", "rules",
    "constraint", "constraints",
    "limit", "limits",
    "range", "ranges",
    "size", "sizes",
    "length", "lengths",
    "count", "counts",
    "total", "totals",
    "sum", "sums",
    "average", "averages",

    # ── Action-verb tokens that are almost certainly METHOD names, not classes ──
    # These appear in scenarios as method descriptions e.g. "deposit()", "withdraw()".
    # The NLP extractor strips "()" and may emit them as capitalised tokens like
    # "Deposit", "Withdraw" — without this list they get falsely flagged as
    # MISSING_CLASS.  If a scenario genuinely uses one as a class name, add it
    # to _DOMAIN_CLASS_WHITELIST inside _is_instruction_word().
    "deposit", "withdraw",
    "calculate", "apply",
    "open", "request",
    "manage", "hold", "maintain", "provide",
    "belong", "associate", "get", "set", "call", "perform",
    # Domain-flavoured adjectives/descriptors that are not class names
    # NOTE: "banking" removed — it is not a class name per se but keep it
    #       here to avoid false MISSING_CLASS; "specialized" is always an adj
    "specialized",

    # ── Compound noise tokens NLP may emit ───────────────────────────────────
    "accountnumber", "contactinfo", "customerid", "loanid", "bankname",
    "openaccount", "requestloan", "calculateinterest", "applyoverdraft",
    "classdiagram", "umlclass", "classobject",
}

# ── Suffixes that strongly indicate a word is NOT a class name ───────────────
# Covers: "classification", "notation", "description", "verification",
#         "drawing", "modelling", "relationship", "assignment", etc.
_NOISE_SUFFIXES: tuple = (
    "tion", "tions",      # classification, notation, association …
    "sion", "sions",      # expression, discussion …
    "ness", "nesses",     # correctness, completeness …
    "ment", "ments",      # assignment, requirement …
    "ance", "ances",      # relevance, importance …
    "ence", "ences",      # reference, difference …
    "ity", "ities",       # multiplicity, visibility …
    "ism", "isms",        # polymorphism …
    "ology", "ologies",   # methodology …
    "ship", "ships",      # relationship …
    "ing", "ings",        # drawing, modelling …
    "ation", "ations",    # generalization, representation …
)

# Regex: a valid class name starts with an UPPERCASE letter (PascalCase)
_VALID_CLASS_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")

def _is_instruction_word(word: str) -> bool:
    """
    Return True if this token should be excluded from scenario_classes.

    Checks (in order):
      1. Empty / whitespace
      2. Known noise word (exact match in _NOISE_WORDS)
      3. Contains spaces / parens / colon / non-alphanumeric → not a class
      4. Starts with lowercase → variable / attribute name
      5. All-uppercase short token ≤ 3 chars (e.g. "ID", "UI", "UML")
      6. Doesn't match PascalCase pattern
      7. Ends with a noise suffix (e.g. "Classification", "Notation", "Drawing")
      8. Too short (≤ 2 chars) to be a meaningful class name
    """
    w = word.strip()
    if not w:
        return True
    wl = w.lower()
    # 1. Known noise word (exact)
    if wl in _NOISE_WORDS:
        return True
    # 2. Contains spaces → phrase, not a class name
    if " " in w:
        return True
    # 3. Contains parentheses → method call
    if "(" in w or ")" in w:
        return True
    # 4. Contains colon → attribute with type annotation
    if ":" in w:
        return True
    # 5. Starts with lowercase → camelCase variable / attribute
    if w[0].islower():
        return True
    # 6. Contains non-alphanumeric characters (e.g. "bank_name", "0..*")
    if re.search(r"[^a-zA-Z0-9]", w):
        return True
    # 7. All-uppercase AND short → acronym, not a domain class
    if w.isupper() and len(w) <= 3:
        return True
    # 8. Must match PascalCase
    if not _VALID_CLASS_RE.match(w):
        return True
    # 9. Ends with a noise suffix → meta / instruction word, not a domain class
    #    e.g. "Classification", "Notation", "Description", "Drawing", "Modelling"
    #    EXCEPTION: known domain class names that happen to end in a noise suffix.
    #    These are commonly used as class names in OOP exercises and must NOT be
    #    treated as noise regardless of their suffix.
    _DOMAIN_CLASS_WHITELIST: Set[str] = {
        # -tion / -sion words that are valid domain classes
        "section", "transaction", "subscription", "reservation",
        "notification", "connection", "collection", "location",
        "station", "position", "condition", "formation",
        "application", "operation", "operations", "permission", "registration",
        "production", "inspection", "submission", "adoption",
        "protection", "promotion", "session", "question",
        # -ment words
        "department", "payment", "shipment", "appointment",
        "enrollment", "enrolment", "statement", "assessment",
        "assignment", "movement", "treatment", "agreement",
        # -ance / -ence words
        "insurance", "attendance", "preference",
        "conference", "reference", "licence", "license",
        # -ness words
        "address", "business",
        # -ship words
        "membership",
        # -ing words
        "booking", "listing", "reading", "meeting", "setting",
        "billing", "rating", "ranking", "posting", "offering",
        # -ity words
        "university", "community", "facility", "entity",
        "authority", "activity", "property",
        # other common OOP class names with noise suffixes
        "member",   "members",   "section",   "sections",
        # ── Dataset-specific compound / suffixed class names ──────────────────
        # These appear in the class diagram dataset (Sections A & B) and would
        # otherwise be filtered out by the noise-suffix check.
        "bloodgroup",        # BloodGroup
        "bloodunit",         # BloodUnit
        "bloodbank",         # BloodBank
        "tourpackage",       # TourPackage
        "travelagency",      # TravelAgency
        "couriercompany",    # CourierCompany
        "serviceplan",       # ServicePlan
        "telecomprovider",   # TelecomProvider
        "reliefcamp",        # ReliefCamp
        "disastersystem",    # DisasterSystem
        "sportsacademy",     # SportsAcademy
        "musicacademy",      # MusicAcademy
        "onlineforum",       # OnlineForum
        "servicerequest",    # ServiceRequest
        "leaverequest",      # LeaveRequest
        "supportagent",      # SupportAgent
        "transportcompany",  # TransportCompany
        "parkingarea",       # ParkingArea
        "shoppingmall",      # ShoppingMall
        "constructioncompany",  # ConstructionCompany
        "researchinstitute", # ResearchInstitute
        "inspectioncenter",  # InspectionCenter
        "insurancecompany",  # InsuranceCompany
        "logisticscompany",  # LogisticsCompany
        "deliveryvan",       # DeliveryVan
        "healthpolicy",      # HealthPolicy
        "vehiclepolicy",     # VehiclePolicy
        "savingsaccount",    # SavingsAccount
        "currentaccount",    # CurrentAccount
        "permanentfaculty",  # PermanentFaculty
        "visitingfaculty",   # VisitingFaculty
        "facultymember",     # FacultyMember
        "undergraduatestudent",  # UndergraduateStudent
        "postgraduatestudent",   # PostgraduateStudent
        "technicalstaff",    # TechnicalStaff
        "administrativestaff",   # AdministrativeStaff
        "transportservice",  # TransportService
        "utilityservice",    # UtilityService
        "digitalmedia",      # DigitalMedia
        "cityemployee",      # CityEmployee
        "receiver",          # Receiver (CourierCompany scenario)
        "victim",            # Victim (DisasterSystem)
        "trainee",           # Trainee (Academy)
        "housekeeper",       # Housekeeper (Hotel housekeeping)
        "publication",       # Publication (ResearchInstitute)
        "publication",
        "subscription",
        "supplier",          # Supplier (Warehouse scenario)
        "inventory",         # Inventory (Warehouse)
        "engineer",          # Engineer (Construction)
        "researcher",        # Researcher
        "librarian",         # Librarian
        "inspection",        # Inspection (vehicle inspection)
        "leavemanagement",   # alias
        # Section B attributes/methods scenario class names
        "payroll",           # Payroll
        "reservation",
        "appointment",
        "feedback",          # Feedback
        "subscription",
        "wallet",            # Wallet
        "review",            # Review
        "voucher",
        "bill",              # Bill
        "quiz",              # Quiz
        "fine",              # Fine (LibraryFine)
        "vote",              # Vote
        "voter",             # Voter
        "ticket",            # Ticket
        "slot",              # Slot (Parking)
        "session",           # Session (Gym)
        "trainer",           # Trainer
        "coach",             # Coach
        "player",            # Player
        "thread",            # Thread (OnlineForum)
        "club",              # Club
        "ward",              # Ward (Hospital)
        "loan",              # Loan
        "account",           # Account
        "claim",             # Claim
        "policy",            # Policy
        "resource",          # Resource (Library)
        "assignment",
        "learner",           # Learner
        "instructor",        # Instructor
        "citizen",           # Citizen
        "buyer",             # Buyer
        "seller",            # Seller
        "product",           # Product
        "parcel",            # Parcel
        "driver",            # Driver
        "route",             # Route
        "vehicle",           # Vehicle
        "bus",               # Bus
        "aircraft",          # Aircraft
        "flight",            # Flight
        "passenger",         # Passenger
        "screen",            # Screen (Cinema)
        "show",              # Show (Cinema)
        "movie",             # Movie
    }
    if wl.endswith(_NOISE_SUFFIXES) and wl not in _DOMAIN_CLASS_WHITELIST:
        return True
    # 10. Too short to be a meaningful class name
    if len(w) <= 2:
        return True
    return False
def _normalize(name: str) -> str:
    return name.strip().lower()

# ══════════════════════════════════════════════════════════════════════════════
#  UML RELATIONSHIP CLASSIFIER
#  Analyses raw scenario text sentence-by-sentence and classifies each
#  X → Y pair into one of four canonical UML relationship types using the
#  trigger-phrase rules supplied in the validator spec:
#
#  Association   – "X has Y", "X uses Y", "X is related to Y"
#  Aggregation   – "X contains Y", "X is a collection of Y",
#                  "Y is a member of X", "X manages Y",
#                  "X holds Y", "X is made up of Y"
#  Composition   – "X is composed of Y", "Y cannot exist without X",
#                  "X strictly owns Y", "X owns Y",
#                  "X consists of Y", "X is part of Y" (when whole destroys part)
#  Generalization– "X is a type of Y", "X is a kind of Y",
#                  "X inherits from Y", "X extends Y",
#                  "X is a Y", "X is an Y",
#                  "SavingsAccount / CurrentAccount / … inherit from Account"
#
#  Returns a list of dicts:
#    { "from": str, "to": str, "type": str, "trigger": str }
#  where "trigger" is the matched phrase for traceability.
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (regex_pattern, rel_type, from_group, to_group)
# Groups 1/2 capture the two entity names from the sentence.
_REL_TRIGGER_PATTERNS: List[tuple] = [
    # ── Composition triggers ──────────────────────────────────────────────────
    # "X is composed of Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+is\s+composed\s+of\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "composition", 1, 2),
    # "Y cannot exist without X"  →  from=Y, to=X
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+cannot\s+exist\s+without\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "composition", 1, 2),
    # "X strictly owns Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+strictly\s+owns\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "composition", 1, 2),
    # "X owns Y"  (weak composition trigger — lower priority than strictly owns)
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+owns\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "composition", 1, 2),
    # "X consists of Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+consists\s+of\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "composition", 1, 2),

    # ── Aggregation triggers ──────────────────────────────────────────────────
    # "X contains Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+contains\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "aggregation", 1, 2),
    # "X is a collection of Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+is\s+a\s+collection\s+of\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "aggregation", 1, 2),
    # "Y is a member of X"  →  from=Y, to=X (Y belongs to X)
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+is\s+a\s+member\s+of\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "aggregation", 1, 2),
    # "X holds Y" / "X may hold Y" — skip optional quantifier words
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+(?:may\s+|can\s+|could\s+)?holds?"
        r"(?:\s+(?:zero\s+or\s+more|one\s+or\s+more|multiple|many|several|various|zero|one|two|three|some|few|a))?"
        r"\s+([A-Z][a-zA-Z0-9]+)",
        re.IGNORECASE), "aggregation", 1, 2),
    # "X is made up of Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+is\s+made\s+up\s+of\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "aggregation", 1, 2),

    # ── Generalization triggers ───────────────────────────────────────────────
    # "X is a type of Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+is\s+a\s+type\s+of\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "generalization", 1, 2),
    # "X is a kind of Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+is\s+a\s+kind\s+of\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "generalization", 1, 2),
    # "X inherits from Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+inherits\s+from\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "generalization", 1, 2),
    # "X extends Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+extends\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "generalization", 1, 2),
    # "X is a Y" / "X is an Y"  — only when Y is a known class-like word
    # (kept broad; caller should filter noise with real_scenario_classes)
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+is\s+an?\s+([A-Z][A-Za-z0-9]*)",
        re.IGNORECASE), "generalization", 1, 2),
    # "SavingsAccount and CurrentAccount inherit from Account"  (multi-child form)
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+inherit(?:s)?\s+from\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "generalization", 1, 2),
    # "X and Y inherit from Z"  — handled via multi-child split below

    # ── Association triggers ──────────────────────────────────────────────────
    # "X has Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+has\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "association", 1, 2),
    # "X uses Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+uses\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "association", 1, 2),
    # "X manages Y"  — supervisory relationship = association (not aggregation)
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+manages"
        r"(?:[\s]+(?:multiple|many|several|various|zero|one|two|three|some|few|a))?"
        r"[\s]+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "association", 1, 2),
    # "X is related to Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+is\s+related\s+to\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "association", 1, 2),
    # "X is associated with Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+is\s+associated\s+with\s+([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "association", 1, 2),
    # "X must be associated with Y" / "X is associated with Y"
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+must\s+be\s+associated\s+with\s+"
        r"(?:one\s+|exactly\s+one\s+|a\s+|an\s+)?([A-Z][a-zA-Z0-9]+)",
        re.IGNORECASE), "association", 1, 2),
    # "X may apply for Y" / "X apply for Y" (e.g. Customers may apply for Loans)
    (re.compile(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+(?:may\s+|can\s+)?apply\s+for\s+"
        r"(?:multiple\s+|many\s+|several\s+|zero\s+or\s+more\s+)?"
        r"([A-Za-z][A-Za-z0-9]*)",
        re.IGNORECASE), "association", 1, 2),
]

# Priority order for de-duplication when multiple patterns match the same pair.
# More specific types (composition > aggregation > generalization > association)
_REL_TYPE_PRIORITY = {
    "composition":    4,
    "aggregation":    3,
    "generalization": 2,
    "association":    1,
}


def classify_scenario_relationships(
    raw_scenario: str,
    known_classes: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Scan every sentence of *raw_scenario* and extract (from, to, type, trigger)
    tuples using the four UML trigger-phrase rule sets.

    Parameters
    ----------
    raw_scenario  : The full scenario text (may be multi-sentence).
    known_classes : Optional set of lowercase class names from the NLP extractor.
                    When supplied, only pairs where BOTH names appear in this set
                    are kept — this filters out noise words like "a", "the", etc.

    Returns
    -------
    List of dicts, one per classified relationship pair, de-duplicated by
    (from, to) with highest-priority type winning.
    """
    # Split into sentences for cleaner matching
    sentences = re.split(r"[.;!\n]+", raw_scenario)

    best: Dict[tuple, Dict[str, Any]] = {}  # (from_n, to_n) → best match

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # ── Multi-child inherit pattern: "A and B inherit from C" ─────────────
        # Capture comma/and-separated list of children before "inherit(s) from X"
        multi_inh = re.search(
            r"([A-Za-z][A-Za-z0-9]*(?:\s*(?:,|and)\s*[A-Za-z][A-Za-z0-9]*)+)"
            r"\s+inherit(?:s)?\s+from\s+([A-Za-z][A-Za-z0-9]*)",
            sent, re.IGNORECASE
        )
        _multi_matched = False
        if multi_inh:
            parent = multi_inh.group(2)
            children_raw = re.split(r"\s*(?:,|and)\s*", multi_inh.group(1))
            for child in children_raw:
                child = child.strip()
                if not child:
                    continue
                _register(best, child, parent, "generalization",
                          f"{child} inherits from {parent}", known_classes)
            _multi_matched = True

        # ── "X and Y provide / are / extend Z" patterns ───────────────────────
        # e.g. "SavingsAccount and CurrentAccount inherit from Account"
        # (already handled above) but also:
        # "SavingsAccount and CurrentAccount extend Account"
        multi_ext = re.search(
            r"([A-Za-z][A-Za-z0-9]*(?:\s*(?:,|and)\s*[A-Za-z][A-Za-z0-9]*)+)"
            r"\s+extend(?:s)?\s+([A-Za-z][A-Za-z0-9]*)",
            sent, re.IGNORECASE
        )
        if multi_ext:
            parent = multi_ext.group(2)
            children_raw = re.split(r"\s*(?:,|and)\s*", multi_ext.group(1))
            for child in children_raw:
                child = child.strip()
                if child:
                    _register(best, child, parent, "generalization",
                              f"{child} extends {parent}", known_classes)
            _multi_matched = True

        if _multi_matched:
            # Single-pair patterns will still run to catch other relationships
            # in the same sentence, but we record that multi-child was handled.
            pass

        # ── Single-pair trigger patterns ──────────────────────────────────────
        for (pattern, rel_type, fg, tg) in _REL_TRIGGER_PATTERNS:
            for m in pattern.finditer(sent):
                frm = m.group(fg).strip()
                to  = m.group(tg).strip()
                trigger = m.group(0)
                _register(best, frm, to, rel_type, trigger, known_classes)

    return list(best.values())


def _register(
    best: Dict[tuple, Dict[str, Any]],
    frm: str,
    to: str,
    rel_type: str,
    trigger: str,
    known_classes: Optional[Set[str]],
) -> None:
    """Insert or upgrade a (frm, to) entry in *best* using type priority."""
    frm = frm.strip()
    to  = to.strip()
    if not frm or not to or frm.lower() == to.lower():
        return
    # Normalize plural to singular so "Customers" → "Customer" when storing
    # This keeps stored names consistent with class labels in the diagram.
    def _depluralize(name: str, classes: Optional[Set[str]]) -> str:
        if not classes:
            return name
        nl = name.lower()
        if nl in classes:
            return name  # already exact
        candidates = []
        if nl.endswith("es") and len(nl) > 4:
            candidates.append((nl[:-2], name[:-2]))
        if nl.endswith("s") and len(nl) > 3:
            candidates.append((nl[:-1], name[:-1]))
        for stem, original in candidates:
            if stem and stem in classes:
                return original  # return de-pluralized with original casing
        return name
    frm = _depluralize(frm, known_classes)
    to  = _depluralize(to,  known_classes)
    # Filter noise: both names must be in known_classes (if supplied).
    # We also try the de-pluralized form (strip trailing 's' or 'es') so
    # "Customers" matches "Customer", "Accounts" matches "Account", etc.
    if known_classes:
        def _class_match(name: str) -> bool:
            nl = name.lower()
            if nl in known_classes:
                return True
            # Fuzzy match only for names long enough to be reliable (≥5 chars)
            if len(nl) >= 5 and any(
                difflib.SequenceMatcher(None, nl, k).ratio() >= 0.82
                for k in known_classes if len(k) >= 4
            ):
                return True
            # Try de-pluralized forms
            stems = []
            if nl.endswith("es") and len(nl) > 4:
                stems.append(nl[:-2])
            if nl.endswith("s") and len(nl) > 3:
                stems.append(nl[:-1])
            for _stem in stems:
                if _stem in known_classes:
                    return True
                if len(_stem) >= 5 and any(
                    difflib.SequenceMatcher(None, _stem, k).ratio() >= 0.82
                    for k in known_classes if len(k) >= 4
                ):
                    return True
            return False

        if not _class_match(frm) or not _class_match(to):
            return
    key = (_normalize(frm), _normalize(to))
    existing = best.get(key)
    new_priority = _REL_TYPE_PRIORITY.get(rel_type, 0)
    if existing is None or new_priority > _REL_TYPE_PRIORITY.get(existing["type"], 0):
        best[key] = {
            "from":    frm,
            "to":      to,
            "type":    rel_type,
            "trigger": trigger,
        }


class ClassDiagramValidator(BaseValidator):
    def validate(
        self,
        extracted: Dict[str, Any],
        shapes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        errors: List[ValidationError] = []
        # ── DEBUG: log raw shape keys ─────────────────────────────────────────
        for _i, _s in enumerate(shapes[:10]):   # first 10 shapes only
            _log.debug(f"[ClassValidator] shape[{_i}] keys={list(_s.keys())} "
                       f"type={_s.get('type','?')} "
                       f"from={_s.get('from', _s.get('source','?'))} "
                       f"to={_s.get('to', _s.get('target','?'))}")
        # ── Parse shapes ──────────────────────────────────────────────────────
        # A shape is a class-box if its type contains any of these keywords
        CLASS_TYPE_KEYWORDS = (
            "class", "rectangle", "rect", "box", "entity", "node",
            "umlclass", "classshape", "class_shape",
            "shape",   # generic fallback
        )
        REL_TYPE_KEYWORDS   = ("arrow", "line", "connect", "association", "relation",
                               "generalization", "inheritance", "aggregation",
                               "composition", "dependency", "link", "edge")

        def _is_class_shape(s):
            t = str(s.get("type", s.get("shapeType", s.get("shape_type", "")))).lower()
            # Explicit relationship types are never classes
            if any(k in t for k in REL_TYPE_KEYWORDS):
                return False
            # If the type explicitly contains a class keyword → class
            if any(k in t for k in CLASS_TYPE_KEYWORDS):
                return True
            # ── Semantic fallback ──────────────────────────────────────────────
            # If the shape has no recognised type BUT has a non-empty text/name
            # AND has no endPosition (i.e. it is not a line/arrow), treat it as
            # a class box.  This covers plain rectangles dropped from the toolbox
            # that arrive with type="" or type="unknown".
            has_label = any(
                str(s.get(k, "")).strip() not in ("", "none", "null", "undefined")
                for k in ("text", "name", "label", "title", "className",
                          "class_name", "caption", "value", "content")
            )
            has_end_position = s.get("endPosition") is not None
            if has_label and not has_end_position:
                return True
            return False
        class_shapes  = [s for s in shapes if _is_class_shape(s)]
        relationships = self._get_relationships(shapes)
        _log.debug(f"[ClassValidator] found {len(class_shapes)} classes, "
                   f"{len(relationships)} relationships: {relationships}")
        def _get_class_label(s):
            # ── Priority: dedicated class-name fields ─────────────────────────
            for key in ("className", "class_name", "header", "title", "name",
                        "label", "caption"):
                v = str(s.get(key, "")).strip()
                if v and v.lower() not in ("none", "null", "undefined", ""):
                    return v
            # ── Fallback: "text" field ─────────────────────────────────────────
            # Flutter's classShape stores the class name in "text".
            # However "text" may sometimes contain multiline content like
            # "ClassName\nattr1\nattr2"; use only the first non-empty line.
            raw_text = str(s.get("text", s.get("value", s.get("content", "")))).strip()
            if raw_text and raw_text.lower() not in ("none", "null", "undefined", ""):
                # Strip section markers that Flutter may inject before the class name
                _SECTION_MARKERS_LBL = ("---attrs---", "---ops---", "---methods---",
                                        "---attributes---", "---operations---")
                raw_lower = raw_text.lower()
                for _marker in _SECTION_MARKERS_LBL:
                    _idx = raw_lower.find(_marker)
                    if _idx != -1:
                        raw_text = raw_text[:_idx].strip()
                        raw_lower = raw_text.lower()
                for line in raw_text.splitlines():
                    line = line.strip()
                    if line and line.lower() not in ("none", "null", "undefined", ""):
                        return line
            return ""
        raw_labels   = [_get_class_label(s) for s in class_shapes]
        class_labels = [lbl for lbl in raw_labels if lbl]
        # ── Extracted from scenario ───────────────────────────────────────────
        scenario_classes = extracted.get("classes", [])
        scenario_rels    = extracted.get("relationships", [])
        # Filter out instruction/noise words from scenario classes
        # Also filter out words that are actually operations/methods from the scenario
        # e.g. diagnosePatient, prescribeMedicine, requestAppointment, viewReport
        # These appear in NLP extracted classes but are methods, not classes.
        _scenario_methods_set: Set[str] = {
            m.lower() for m in extracted.get("methods", [])
        }
        _scenario_actions_set: Set[str] = {
            a.lower() for a in extracted.get("actions", [])
        }
        # Also detect camelCase compound words that look like method names
        # e.g. "DiagnosePatient" → has verb prefix like diagnose, prescribe, request, view
        _METHOD_VERB_PREFIXES = (
            "diagnose", "prescribe", "request", "view", "login", "logout",
            "register", "calculate", "process", "validate", "create", "delete",
            "update", "fetch", "send", "receive", "submit", "check", "verify",
            "display", "render", "handle", "perform", "execute", "run",
            "add", "remove", "list", "search", "filter", "sort", "load",
            "save", "store", "retrieve", "upload", "download", "export",
            "import", "print", "format", "parse", "convert", "generate",
            "assign", "schedule", "cancel", "confirm", "approve", "reject",
        )
        def _looks_like_method(word: str) -> bool:
            """Return True if this word looks like a method/operation name, not a class."""
            wl = word.lower()
            # If NLP already classified it as a method/action, exclude it
            if wl in _scenario_methods_set or wl in _scenario_actions_set:
                return True
            # Detect verb-prefixed compound words (camelCase or PascalCase method names)
            # e.g. "DiagnosePatient", "PrescribeMedicine", "RequestAppointment"
            for prefix in _METHOD_VERB_PREFIXES:
                if wl.startswith(prefix) and len(wl) > len(prefix):
                    # The rest should be a noun (starts with uppercase in PascalCase)
                    rest = word[len(prefix):]
                    if rest and rest[0].isupper():
                        return True
            return False

        real_scenario_classes = [
            c for c in scenario_classes
            if not _is_instruction_word(c) and not _looks_like_method(c)
        ]

        # ── Bug-1 fix: Raw PascalCase scanner ────────────────────────────────
        # NLP extractors sometimes miss domain class names that appear in the
        # raw scenario text (e.g. "Operation" when the scenario says "the system
        # records each banking operation").  We scan the raw scenario for any
        # PascalCase token that:
        #   (a) is NOT an instruction/noise word
        #   (b) is NOT already in real_scenario_classes (case-insensitive)
        # and add it.  This ensures classes like "Operation" are not silently
        # skipped just because the NLP extractor omitted them.
        _raw_for_scan = extracted.get("scenario", extracted.get("raw_text", ""))
        if not _raw_for_scan:
            _raw_for_scan = " ".join(str(v) for v in extracted.values() if isinstance(v, str))
        _existing_sc_lower = {c.lower() for c in real_scenario_classes}
        for _tok in re.findall(r"[A-Z][a-z][a-zA-Z0-9]*", _raw_for_scan):
            if _tok.lower() not in _existing_sc_lower and not _is_instruction_word(_tok) and not _looks_like_method(_tok):
                real_scenario_classes.append(_tok)
                _existing_sc_lower.add(_tok.lower())
        # Does scenario mention any relationships?
        # Also true when _global_expected_rel_type is detected (set below), but
        # we set a provisional flag here and update it after the detection block.
        scenario_has_relationships = bool(scenario_rels) or bool(
            extracted.get("associations", [])
        )
        # Build scenario word set for fuzzy spelling checks (R11)
        scenario_words = _extract_scenario_words(extracted)

        # ── UML Relationship Classifier ───────────────────────────────────────
        # Run the sentence-level trigger classifier on the raw scenario text.
        # This produces per-pair relationship types using the four rule sets:
        #   Association    → "X has Y", "X uses Y", "X is related to Y"
        #   Aggregation    → "X contains Y", "X is a collection of Y",
        #                    "Y is a member of X", "X manages Y", "X holds Y"
        #   Composition    → "X is composed of Y", "Y cannot exist without X",
        #                    "X strictly owns Y", "X owns Y"
        #   Generalization → "X is a type of Y", "X is a kind of Y",
        #                    "X inherits from Y", "X extends Y"
        #
        # Classified pairs AUGMENT (never replace) scenario_rels so validation
        # works even when the NLP extractor returns no per-pair entries.
        _raw_scenario_full = extracted.get("scenario", extracted.get("raw_text", ""))
        if not _raw_scenario_full:
            # Fallback: concatenate all string values from the extracted dict
            # (older NLP extractor versions that don't include "scenario" key)
            _raw_scenario_full = " ".join(
                str(v) for v in extracted.values() if isinstance(v, str)
            )
        _known_cls_set: Set[str] = {c.lower() for c in real_scenario_classes}
        _classifier_rels: List[Dict[str, Any]] = classify_scenario_relationships(
            _raw_scenario_full,
            known_classes=_known_cls_set if _known_cls_set else None,
        )
        _log.debug(f"[ClassValidator] classifier_rels={_classifier_rels}")

        # Merge: only add pairs NOT already covered by the NLP extractor
        _existing_pairs: Set[tuple] = set()
        for _sr in scenario_rels:
            _sf = _normalize(str(_sr.get("from", _sr.get("source", ""))))
            _st = _normalize(str(_sr.get("to",   _sr.get("target", ""))))
            if _sf and _st:
                _existing_pairs.add((_sf, _st))
                _existing_pairs.add((_st, _sf))

        for _cr in _classifier_rels:
            _cf = _normalize(_cr["from"])
            _ct = _normalize(_cr["to"])
            if (_cf, _ct) not in _existing_pairs and (_ct, _cf) not in _existing_pairs:
                scenario_rels.append({
                    "from":    _cr["from"],
                    "to":      _cr["to"],
                    "type":    _cr["type"],
                    "trigger": _cr.get("trigger", ""),
                })
                _existing_pairs.add((_cf, _ct))
                _existing_pairs.add((_ct, _cf))

        # Refresh flag — classifier may have found new relationships
        if _classifier_rels:
            scenario_has_relationships = True

        # ── Detect global expected relationship type from scenario text ────────
        # This is the KEY fallback for two important cases:
        #
        # Case A: Scenario says "generalization relationship" explicitly →
        #   NLP puts individual class names in scenario_rels (if at all) but
        #   may not emit per-pair entries.  We detect the keyword globally.
        #
        # Case B: Scenario says "animal having duck fish zebra" with NO explicit
        #   relationship word → IS-A (generalization) is the correct UML type
        #   because duck/fish/zebra ARE animals.  We detect this via semantic
        #   is-a heuristics (parent class name appears in diagram, children are
        #   known animal subtypes, or the scenario lists classes that are
        #   real-world subtypes of the first/main class).
        #
        # Used by: R_GLOBAL_WRONG_TYPE block and S3-GLOBAL block (both below).
        # ─────────────────────────────────────────────────────────────────────
        _RAW_SCENARIO_TEXT = " ".join(
            str(v) for v in extracted.values() if isinstance(v, str)
        ).lower()
        for _v in extracted.values():
            if isinstance(_v, list):
                _RAW_SCENARIO_TEXT += " " + " ".join(str(x) for x in _v).lower()

        _GLOBAL_GEN_KEYWORDS = (
            "generalization", "generalisation", "inheritance", "inherits",
            "inherit", "extends", "is-a", "isa", "parent", "child",
            "superclass", "subclass", "derived", "base class", "parent class",
            "sub class", "subtype", "super type",
        )
        _GLOBAL_ASSOC_KEYWORDS  = ("association", "associated with", "associates")
        _GLOBAL_AGG_KEYWORDS    = ("aggregation", "aggregates", "is a collection of", "is made up of")
        _GLOBAL_COMP_KEYWORDS   = ("composition", "composed of", "consists of", "strictly owns")
        _GLOBAL_DEP_KEYWORDS    = ("dependency", "depends on", "dependent")

        _global_expected_rel_type: str = ""
        if any(k in _RAW_SCENARIO_TEXT for k in _GLOBAL_GEN_KEYWORDS):
            _global_expected_rel_type = "generalization"
        elif any(k in _RAW_SCENARIO_TEXT for k in _GLOBAL_COMP_KEYWORDS):
            _global_expected_rel_type = "composition"
        elif any(k in _RAW_SCENARIO_TEXT for k in _GLOBAL_AGG_KEYWORDS):
            _global_expected_rel_type = "aggregation"
        elif any(k in _RAW_SCENARIO_TEXT for k in _GLOBAL_ASSOC_KEYWORDS):
            _global_expected_rel_type = "association"
        elif any(k in _RAW_SCENARIO_TEXT for k in _GLOBAL_DEP_KEYWORDS):
            _global_expected_rel_type = "dependency"

        # Update scenario_has_relationships: if a global type was detected,
        # the scenario does imply relationships even if scenario_rels is empty.
        if _global_expected_rel_type:
            scenario_has_relationships = True

        # ── Implicit IS-A heuristic (Case B) ──────────────────────────────────
        # When NO relationship keyword appears in the scenario at all, check
        # whether the classes listed are real-world IS-A subtypes of the first
        # (parent) class.  If yes, assume generalization is expected.
        # Heuristic: if scenario has ≥1 class whose name is a well-known
        # subtype/instance of another class in the same scenario, mark as
        # generalization.  We keep this intentionally broad for common animal/
        # vehicle/shape/person hierarchies taught in OOP courses.
        if not _global_expected_rel_type and real_scenario_classes:
            # Known IS-A parent→child mappings (extend as needed)
            _ISA_HEURISTICS: Dict[str, Set[str]] = {
                "animal":   {"duck","fish","zebra","dog","cat","bird","lion","tiger",
                             "horse","cow","snake","whale","shark","eagle","parrot",
                             "rabbit","fox","wolf","bear","elephant","monkey","frog",
                             "penguin","crocodile","turtle","deer","goat","sheep"},
                "vehicle":  {"car","truck","bus","bike","motorcycle","bicycle",
                             "train","plane","airplane","boat","ship","van","scooter"},
                "shape":    {"circle","square","rectangle","triangle","polygon",
                             "oval","ellipse","hexagon","pentagon"},
                "person":   {"student","teacher","employee","manager","doctor",
                             "nurse","admin","administrator","customer","user",
                             "professor","instructor","staff","worker","client"},
                "account":  {"savingsaccount","currentaccount","checkingaccount",
                             "loanaccount","bankaccount","creditaccount"},
                "employee": {"manager","developer","engineer","designer","intern",
                             "director","supervisor"},
            }
            _sc_lower = {c.lower() for c in real_scenario_classes}
            for _parent, _children in _ISA_HEURISTICS.items():
                if _parent in _sc_lower and _children & _sc_lower:
                    _global_expected_rel_type = "generalization"
                    break
        # ═════════════════════════════════════════════════════════════════════
        #  R8 — Empty / missing class name
        #  Uses "Class 1", "Class 2" numbering (1-based)
        # ═════════════════════════════════════════════════════════════════════
        for idx, lbl in enumerate(raw_labels):
            if not lbl:
                class_num = idx + 1
                errors.append(ValidationError(
                    error_type  = "EMPTY_CLASS",
                    description = f"Class {class_num} has no name.",
                    suggestion  = "Every class must have a name. Add a meaningful name to the class.",
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = f"Class {class_num}",
                ))
        # ═════════════════════════════════════════════════════════════════════
        #  R1 — Missing classes (scenario → diagram)
        #  Flexible: filters instruction words, uses fuzzy match.
        #  NEW: If a class label is 50-70% similar to a scenario class
        #       (close but not exact), report it as a SPELLING MISTAKE
        #       instead of a missing class.
        # ═════════════════════════════════════════════════════════════════════
        for cls in real_scenario_classes:
            cls_lower = cls.lower()
            # Step 1: Exact / strong match (>=92%) → class is correct, skip.
            # 92% allows for trivial casing/one-char differences (e.g. "Accounts"
            # vs "Account") while flagging real spelling mistakes like "Custumer".
            exact_match = any(
                difflib.SequenceMatcher(None, cls_lower, lbl.lower()).ratio() >= 0.92
                for lbl in class_labels
            )
            if exact_match:
                continue
            # Step 2: Partial similarity (70-91%) → treat as SPELLING MISTAKE
            spelling_match = None
            best_ratio = 0.0
            for lbl in class_labels:
                ratio = difflib.SequenceMatcher(None, cls_lower, lbl.lower()).ratio()
                if 0.70 <= ratio < 0.92 and ratio > best_ratio:
                    best_ratio = ratio
                    spelling_match = lbl
            if spelling_match:
                errors.append(ValidationError(
                    error_type  = "SPELLING_ERROR",
                    description = (
                        f"Spelling mistake: '{spelling_match}' should be '{cls}'. "
                        f"Try changing '{spelling_match}' to '{cls}'."
                    ),
                    suggestion  = f"Rename '{spelling_match}' to '{cls}' to match the scenario.",
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = spelling_match,
                ))
            else:
                # Step 3: No similarity at all → truly missing class
                # Bug-4 fix: if the token looks like an attribute/method name
                # (camelCase, or ends with a known attribute suffix), suggest
                # adding an attribute instead of drawing a whole new class.
                _attr_suffixes = (
                    "id", "Id", "ID",
                    "name", "Name",
                    "number", "Number",
                    "info", "Info",
                    "amount", "Amount",
                    "balance", "Balance",
                    "date", "Date",
                    "type", "Type",
                    "status", "Status",
                    "address", "Address",
                    "phone", "Phone",
                    "email", "Email",
                )
                _looks_like_attribute = (
                    # camelCase: starts uppercase but has lowercase then uppercase (compound)
                    bool(re.match(r'^[A-Z][a-z]+[A-Z]', cls)) or
                    cls.endswith(_attr_suffixes)
                )
                if _looks_like_attribute:
                    errors.append(ValidationError(
                        error_type  = "MISSING_CLASS",
                        description = f"'{cls}' is missing from the diagram.",
                        suggestion  = (
                            f"Add attribute '{cls}' to the relevant class — "
                            f"it looks like a field/attribute, not a separate class."
                        ),
                        severity    = ValidationError.SEVERITY_ERROR,
                        element     = cls,
                    ))
                else:
                    errors.append(ValidationError(
                        error_type  = "MISSING_CLASS",
                        description = f"Class '{cls}' is missing from the diagram.",
                        suggestion  = f"Add class '{cls}' to the diagram.",
                        severity    = ValidationError.SEVERITY_ERROR,
                        element     = cls,
                    ))
        # ── R11 / R12 / R13 / R14 (attributes & methods) — DISABLED ──────────
        # Attribute and method checks are intentionally skipped.
        # Any shape with a class name is treated as a valid class regardless
        # of whether it has attributes, methods, visibility prefixes, or types.
        # ═════════════════════════════════════════════════════════════════════
        #  R4 — Relationship endpoint validation
        #  Only runs if user has actually drawn relationships
        # ═════════════════════════════════════════════════════════════════════
        # Filter out any relationships with unresolved endpoints before all checks
        relationships = [
            r for r in relationships
            if r.get("from","").strip() not in ("", "?", "none", "null")
            and r.get("to","").strip()   not in ("", "?", "none", "null")
        ]
        # Build early id_to_label for R4 checks below
        _id_to_lbl_r4: Dict[str, str] = {}
        for s in shapes:
            sid = str(s.get("id", s.get("shapeId", s.get("nodeId", "")))).strip()
            lbl = _get_class_label(s)
            if sid and lbl:
                _id_to_lbl_r4[sid.lower()] = lbl

        def _resolve_r4(raw: str) -> str:
            r = raw.strip()
            return _id_to_lbl_r4.get(r.lower(), r)

        if relationships:
            for rel in relationships:
                frm_raw = rel.get("from", "")
                to_raw  = rel.get("to", "")
                frm = _resolve_r4(frm_raw)
                to  = _resolve_r4(to_raw)
                if frm and not self._fuzzy_match(frm, class_labels + real_scenario_classes):
                    errors.append(ValidationError(
                        error_type  = "INVALID_RELATIONSHIP",
                        description = f"Relationship source '{frm}' does not match any known class.",
                        suggestion  = f"Verify that '{frm}' exists in the diagram.",
                        severity    = ValidationError.SEVERITY_ERROR,
                        element     = frm,
                    ))
                if to and not self._fuzzy_match(to, class_labels + real_scenario_classes):
                    errors.append(ValidationError(
                        error_type  = "INVALID_RELATIONSHIP",
                        description = f"Relationship target '{to}' does not match any known class.",
                        suggestion  = f"Verify that '{to}' exists in the diagram.",
                        severity    = ValidationError.SEVERITY_ERROR,
                        element     = to,
                    ))
                # R15: Multiplicity format validation
                # Skip for relationship types that never carry multiplicity
                rel_type_r15 = rel.get("type", "").lower()
                _no_mult = ("generalization","inheritance","extends","inherit",
                            "realization","realizes","implements","dependency","depend")
                if not any(k in rel_type_r15 for k in _no_mult):
                    for mult_key in ("multiplicity_from", "multiplicity_to", "multiplicity"):
                        mult = rel.get(mult_key, "").strip()
                        if mult and not VALID_MULTIPLICITY_RE.match(mult):
                            errors.append(ValidationError(
                                error_type  = "INVALID_MULTIPLICITY",
                                description = f"Multiplicity '{mult}' on relationship '{frm}' → '{to}' is not valid.",
                                suggestion  = "Use a valid multiplicity: '1', '0..1', '1..*', '*', '0..*', or a specific number.",
                                severity    = ValidationError.SEVERITY_ERROR,
                                element     = f"{frm} → {to}",
                            ))
        # ═════════════════════════════════════════════════════════════════════
        #  R5 — Extra classes (in diagram, not in scenario)
        # ═════════════════════════════════════════════════════════════════════
        for cls in class_labels:
            if not self._fuzzy_match(cls, real_scenario_classes):
                errors.append(ValidationError(
                    error_type  = "EXTRA_CLASS",
                    description = f"Class '{cls}' is not mentioned in the scenario. Consider removing it.",
                    suggestion  = f"Remove '{cls}' if it is not required by the scenario.",
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = cls,
                ))
        # ═════════════════════════════════════════════════════════════════════
        #  R15b — Missing multiplicity
        #  Only applies to relationships that CARRY multiplicity in UML:
        #    ✓ association, aggregation, composition
        #    ✗ generalization / inheritance / realization / dependency
        #       — these NEVER have multiplicity labels in UML.
        # ═════════════════════════════════════════════════════════════════════

        # Relationship types that NEVER carry multiplicity
        NO_MULTIPLICITY_TYPES = (
            "generalization", "inheritance", "extends", "inherit",
            "realization",    "realizes",    "implements",
            "dependency",     "depend",
        )

        # Relationship types that DO require multiplicity (explicit whitelist)
        NEEDS_MULTIPLICITY_TYPES = (
            "association", "aggregation", "composition",
        )

        def _needs_multiplicity(rel: dict) -> bool:
            """
            Return True ONLY if this relationship type requires multiplicity.

            Design principle:
            - Generalization / inheritance / realization / dependency  -> NEVER need multiplicity.
            - Association / aggregation / composition (explicit type)  -> ALWAYS need multiplicity
              on BOTH ends. Multiplicity is mandatory regardless of scenario context.
            - Generic / unlabeled line (type is empty, "line", "arrow", etc.)
                -> Check scenario_rels for this class pair:
                    - scenario says generalization/dependency -> NO multiplicity.
                    - scenario says association/aggregation/composition -> YES (multiplicity required).
                    - Global scenario expects association/aggregation/composition -> YES.
                    - No scenario entry for this pair and no global type hint
                      -> SKIP (cannot determine intent).
            """
            rtype = rel.get("type", "").lower().strip()

            # 1. Explicit no-multiplicity type
            if any(k in rtype for k in NO_MULTIPLICITY_TYPES):
                return False

            # 2. Explicit needs-multiplicity type (association / aggregation / composition)
            # GUARD: if the scenario globally expects a no-multiplicity type
            # (e.g. generalization/inheritance), do NOT require multiplicity even
            # if the user explicitly drew "association".  The wrong-type error
            # (R_GLOBAL_WRONG_TYPE / R_REL) will already flag that mistake.
            _is_explicit_needs_mult = (
                any(k in rtype for k in NEEDS_MULTIPLICITY_TYPES)
                and not any(k in rtype for k in NO_MULTIPLICITY_TYPES)
            )
            if _is_explicit_needs_mult:
                if _global_expected_rel_type in ("generalization", "dependency"):
                    return False
                # Multiplicity is ALWAYS required on both ends for explicit
                # association / aggregation / composition arrows.
                return True

            # 3. Generic / unlabeled line -> consult scenario_rels first, then global hint
            generic_types_set = {"", "line", "edge", "link", "connect",
                                  "arrow", "connector"}
            if not rtype or rtype in generic_types_set:
                frm_n = _normalize(rel.get("from", ""))
                to_n  = _normalize(rel.get("to",   ""))

                # Per-pair lookup in scenario_rels
                # Use fuzzy matching (same as R_MISSING_REL) so that slight
                # normalisation differences (plural, casing) don't cause misses.
                _found_pair = False
                for sr in scenario_rels:
                    sf    = _normalize(str(sr.get("from",   sr.get("source", ""))))
                    st    = _normalize(str(sr.get("to",     sr.get("target", ""))))
                    stype = str(sr.get("type", sr.get("relation", ""))).lower().strip()
                    # Try exact match first, then fuzzy (≥82 % similarity)
                    def _fm(a: str, b: str) -> bool:
                        if a == b:
                            return True
                        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.82
                    if (_fm(sf, frm_n) and _fm(st, to_n)) or (_fm(sf, to_n) and _fm(st, frm_n)):
                        _found_pair = True
                        if any(k in stype for k in NO_MULTIPLICITY_TYPES):
                            return False
                        if any(k in stype for k in NEEDS_MULTIPLICITY_TYPES):
                            return True
                        # stype is empty or unknown — fall through to global check
                        break

                # No conclusive per-pair entry found — fall back to global expected type
                if _global_expected_rel_type in NEEDS_MULTIPLICITY_TYPES:
                    # Scenario globally expects association/aggregation/composition
                    # → this unlabeled line should carry multiplicity
                    return True
                if _global_expected_rel_type in ("generalization", "dependency"):
                    return False

                # If pair was found in scenario_rels with unknown/empty type,
                # but scenario has relationships at all → require multiplicity
                # (better to show the error than silently skip it)
                if _found_pair and scenario_rels:
                    return True

                # Cannot determine intent (no scenario context at all) -> skip
                return False

            # 4. Any other unknown named type -> skip
            return False

        # Build id_to_label map for resolving node IDs → class names
        id_to_label_mult: Dict[str, str] = {}
        for s in shapes:
            sid = str(s.get("id", s.get("shapeId", s.get("nodeId", "")))).strip()
            lbl = _get_class_label(s)
            if sid and lbl:
                id_to_label_mult[sid.lower()] = lbl

        def _display_name(raw: str) -> str:
            """Return human-readable class name for node ID or label."""
            r = raw.strip()
            resolved = id_to_label_mult.get(r.lower(), "")
            return resolved if resolved else r

        # Pre-build a set of (frm, to) pairs where scenario expects a
        # no-multiplicity relationship (generalization, dependency, etc.).
        # These pairs must be EXCLUDED from the multiplicity check entirely,
        # regardless of what the user actually drew (association, generic line, etc.).
        # This prevents "missing multiplicity" errors when the real problem is
        # "wrong relationship type — draw generalization instead."
        _REL_TYPE_MAP_LOCAL = {
            "association":    ["association", "line"],
            "generalization": ["generalization", "inheritance", "extends", "inherit"],
            "aggregation":    ["aggregation"],
            "composition":    ["composition"],
            "dependency":     ["dependency", "dashed", "uses"],
        }
        _skip_mult_pairs: set = set()
        if scenario_rels:
            for _sr in scenario_rels:
                _sf    = _normalize(str(_sr.get("from",   _sr.get("source", ""))))
                _st    = _normalize(str(_sr.get("to",     _sr.get("target", ""))))
                _stype = str(_sr.get("type", _sr.get("relation", ""))).lower().strip()
                if _sf and _st and any(k in _stype for k in NO_MULTIPLICITY_TYPES):
                    _skip_mult_pairs.add((_sf, _st))
                    _skip_mult_pairs.add((_st, _sf))  # both directions
                    
        # Also detect wrong-type pairs from drawn relationships:
        # if user drew "association" but scenario expects "generalization",
        # add those pairs to _skip_mult_pairs so multiplicity is never checked.
        if scenario_rels and relationships:
            _exp_map_pre: dict = {}
            for _sr in scenario_rels:
                _sf    = _normalize(str(_sr.get("from", _sr.get("source", ""))))
                _st    = _normalize(str(_sr.get("to",   _sr.get("target", ""))))
                _stype = str(_sr.get("type", _sr.get("relation", ""))).lower().strip()
                if _sf and _st:
                    _exp_map_pre[(_sf, _st)] = _stype
                    _exp_map_pre[(_st, _sf)] = _stype  # bidirectional
            for _rel in relationships:
                _rf = _normalize(_rel.get("from", ""))
                _rt = _normalize(_rel.get("to",   ""))
                _rtype = _rel.get("type", "").lower().strip()
                _exp = _exp_map_pre.get((_rf, _rt), "")
                if _exp and any(k in _exp for k in NO_MULTIPLICITY_TYPES):
                    # Scenario expects no-multiplicity type for this pair
                    # → skip multiplicity regardless of drawn type
                    _skip_mult_pairs.add((_rf, _rt))
                    _skip_mult_pairs.add((_rt, _rf))

        # ═════════════════════════════════════════════════════════════════
        #  Best-effort scenario-based multiplicity / label guessing.
        #  Conservative: only returns a guess when the scenario gives an
        #  explicit cardinality/verb cue near both class names — otherwise
        #  returns generic placeholders so we never invent facts.
        # ═════════════════════════════════════════════════════════════════
        _scenario_text_for_guess = extracted.get("scenario", extracted.get("raw_text", "")) or ""

        def _guess_multiplicity_from_scenario(frm: str, to: str) -> tuple:
            """Returns (from_mult, to_mult, confident: bool)."""
            text = _scenario_text_for_guess.lower()
            frm_l, to_l = frm.lower(), to.lower()
            import re as _re_mult
            for sent in _re_mult.split(r'(?<=[.!?])\s+', text):
                if frm_l in sent and to_l in sent:
                    many_words = ("many", "multiple", "several", "various")
                    one_words  = ("one ", "a single", "each ", "single ")
                    # crude ordering: whichever noun a "many"-word appears closer
                    # to on the same side is treated as the "*" end.
                    frm_idx = sent.find(frm_l)
                    to_idx  = sent.find(to_l)
                    frm_many = any(w in sent[:max(frm_idx, 0)+len(frm_l)+15] for w in many_words)
                    to_many  = any(w in sent[max(to_idx-15, 0):to_idx+len(to_l)] for w in many_words)
                    if frm_many and not to_many:
                        return ("*", "1", True)
                    if to_many and not frm_many:
                        return ("1", "*", True)
                    if any(w in sent for w in one_words) and any(w in sent for w in many_words):
                        return ("1", "*", True)
            return ("1", "*", False)  # safe generic default, not confident

        def _guess_relationship_label_from_scenario(frm: str, to: str) -> str:
            """Best-effort verb-based label guess, reusing the same verb list
            used elsewhere for MISSING_ASSOCIATION_LABEL."""
            text = _scenario_text_for_guess.lower()
            frm_l, to_l = frm.lower(), to.lower()
            _verbs = ("manages", "holds", "owns", "provides", "maintains", "employs",
                      "handles", "processes", "serves", "assigns", "registers",
                      "tracks", "records", "stores", "issues", "generates",
                      "creates", "enrolls", "supervises", "coordinates", "schedules",
                      "contains", "has")
            import re as _re_lbl
            for sent in _re_lbl.split(r'(?<=[.!?])\s+', text):
                if frm_l in sent and to_l in sent:
                    for v in _verbs:
                        if f" {v} " in f" {sent} ":
                            return v
            return ""

        if relationships:
            for rel in relationships:
                # ── Skip relationship types that never carry multiplicity ──────
                if not _needs_multiplicity(rel):
                    continue

                frm_raw  = rel.get("from", "")
                to_raw   = rel.get("to", "")
                frm_n    = _normalize(frm_raw)
                to_n     = _normalize(to_raw)
                frm_disp = _display_name(frm_raw)
                to_disp  = _display_name(to_raw)

                # ── NEW GUARD: skip if scenario expects generalization here ───
                # This fires when the user drew "association" (explicit type)
                # but the scenario says generalization — wrong type, not missing
                # multiplicity. The WRONG_RELATIONSHIP_TYPE error handles this.
                if (frm_n, to_n) in _skip_mult_pairs:
                    continue

                # ── Multiplicity check for both explicit and generic relationship types ──
                # Explicit association/aggregation/composition → always require multiplicity.
                # Generic/unlabeled lines → require multiplicity when:
                #   (a) the pair is in scenario_rels, OR
                #   (b) user already added multiplicity on one end (partial → must complete both), OR
                #   (c) scenario globally expects a multiplicity-carrying type
                rtype_drawn = rel.get("type", "").lower().strip()
                _is_explicit_mult = (
                    any(k in rtype_drawn for k in ("association", "aggregation", "composition"))
                    and not any(k in rtype_drawn for k in
                        ("generalization","inheritance","extends","inherit",
                         "realization","dependency","depend"))
                )
                _generic_line_types  = {"line", "edge", "link", "connect",
                                        "arrow", "connector", ""}
                _is_generic = not rtype_drawn or rtype_drawn in _generic_line_types
                if _is_explicit_mult:
                    # Explicit relationship type → multiplicity is MANDATORY on both ends.
                    pass
                elif _is_generic:
                    _pair_in_scenario = any(
                        (_normalize(str(sr.get("from", sr.get("source","")))) == frm_n and
                         _normalize(str(sr.get("to",   sr.get("target","")))) == to_n)
                        or
                        (_normalize(str(sr.get("from", sr.get("source","")))) == to_n and
                         _normalize(str(sr.get("to",   sr.get("target","")))) == frm_n)
                        for sr in scenario_rels
                    ) if scenario_rels else False
                    # If user already drew one multiplicity end, they clearly intended both
                    _partial_mult = bool(
                        rel.get("multiplicity_from", "").strip() or
                        rel.get("multiplicity_to",   "").strip()
                    )
                    _global_needs_mult = _global_expected_rel_type in NEEDS_MULTIPLICITY_TYPES
                    if not _pair_in_scenario and not _partial_mult and not _global_needs_mult:
                        continue

                mult_from = rel.get("multiplicity_from", "").strip()
                mult_to   = rel.get("multiplicity_to",   "").strip()
                if not mult_from and not mult_to:
                    _gf, _gt, _confident = _guess_multiplicity_from_scenario(frm_disp, to_disp)
                    _hint = (f" Based on the scenario, this looks like '{_gf}' on the '{frm_disp}' "
                             f"side and '{_gt}' on the '{to_disp}' side."
                             if _confident else
                             f" Suggested values: '{_gf}' on '{frm_disp}', '{_gt}' on '{to_disp}' "
                             f"— adjust to match your scenario.")
                    errors.append(ValidationError(
                        error_type  = "MISSING_MULTIPLICITY",
                        description = f"Relationship '{frm_disp}' → '{to_disp}' is missing multiplicity on both ends.",
                        suggestion  = ("Add multiplicity labels (e.g., '1', '0..*', '1..*') on both "
                                       "sides of the relationship." + _hint),
                        severity    = ValidationError.SEVERITY_ERROR,
                        element     = f"{frm_disp} → {to_disp}",
                        auto_fix    = {"fixable": True, "action": "add_label",
                                       "from_element": frm_disp, "to_element": to_disp,
                                       "multiplicity_from": _gf, "multiplicity_to": _gt},
                    ))
                elif not mult_from:
                    _gf, _gt, _confident = _guess_multiplicity_from_scenario(frm_disp, to_disp)
                    _hint = f" Suggested: '{_gf}' on the '{frm_disp}' side." if _gf else ""
                    errors.append(ValidationError(
                        error_type  = "MISSING_MULTIPLICITY",
                        description = f"Relationship '{frm_disp}' → '{to_disp}' is missing multiplicity on the '{frm_disp}' side.",
                        suggestion  = (f"Add a multiplicity label (e.g., '1', '0..*') on the "
                                       f"'{frm_disp}' side." + _hint),
                        severity    = ValidationError.SEVERITY_ERROR,
                        element     = f"{frm_disp} → {to_disp}",
                        auto_fix    = {"fixable": True, "action": "add_label",
                                       "from_element": frm_disp, "to_element": to_disp,
                                       "multiplicity_from": _gf, "multiplicity_to": mult_to},
                    ))
                elif not mult_to:
                    _gf, _gt, _confident = _guess_multiplicity_from_scenario(frm_disp, to_disp)
                    _hint = f" Suggested: '{_gt}' on the '{to_disp}' side." if _gt else ""
                    errors.append(ValidationError(
                        error_type  = "MISSING_MULTIPLICITY",
                        description = f"Relationship '{frm_disp}' → '{to_disp}' is missing multiplicity on the '{to_disp}' side.",
                        suggestion  = (f"Add a multiplicity label (e.g., '1', '0..*') on the "
                                       f"'{to_disp}' side." + _hint),
                        severity    = ValidationError.SEVERITY_ERROR,
                        element     = f"{frm_disp} → {to_disp}",
                        auto_fix    = {"fixable": True, "action": "add_label",
                                       "from_element": frm_disp, "to_element": to_disp,
                                       "multiplicity_from": mult_from, "multiplicity_to": _gt},
                    ))

        # ═════════════════════════════════════════════════════════════════
        #  NEW RULE — Association Name is required (unconditional).
        #  Every association/aggregation/composition line must have a label
        #  at its midpoint, regardless of whether the scenario names one or
        #  whether multiplicity applies. Runs in its own loop (NOT gated by
        #  _needs_multiplicity) so it fires even for pairs where multiplicity
        #  requirement couldn't be determined — this is intentionally
        #  separate from MISSING_ASSOCIATION_LABEL above, which only fires
        #  when the scenario explicitly names the expected label.
        # ═════════════════════════════════════════════════════════════════
        if relationships:
            for rel in relationships:
                rtype_chk = rel.get("type", "").lower().strip()
                if any(k in rtype_chk for k in NO_MULTIPLICITY_TYPES):
                    continue  # generalization/dependency/etc. never carry a name label
                if str(rel.get("label", "")).strip():
                    continue  # label present — fine

                frm_disp = _display_name(rel.get("from", ""))
                to_disp  = _display_name(rel.get("to", ""))
                _guess_lbl = _guess_relationship_label_from_scenario(frm_disp, to_disp)
                _suggestion = (
                    f"Add a name label at the midpoint of the line between "
                    f"'{frm_disp}' and '{to_disp}'"
                    + (f" — the scenario suggests '{_guess_lbl}'." if _guess_lbl else ".")
                )
                errors.append(ValidationError(
                    error_type  = "MISSING_ASSOCIATION_NAME",
                    description = "Association Name is required",
                    suggestion  = _suggestion,
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = f"{frm_disp} → {to_disp}",
                    auto_fix    = {"fixable": bool(_guess_lbl), "action": "add_label",
                                   "from_element": frm_disp, "to_element": to_disp,
                                   "name": _guess_lbl},
                ))
        #  Only triggered when scenario itself defines expected relationships
        # ═════════════════════════════════════════════════════════════════════
        if scenario_has_relationships and relationships:
            scenario_all_words = [
                _normalize(c)
                for c in real_scenario_classes + extracted.get("actors", [])
            ]
            for rel in relationships:
                frm = rel.get("from", "").lower()
                to  = rel.get("to", "").lower()
                if frm and to:
                    frm_match = any(
                        re.search(r'\b' + re.escape(frm) + r'\b', w) or
                        re.search(r'\b' + re.escape(w) + r'\b', frm)
                        for w in scenario_all_words
                    )
                    to_match  = any(
                        re.search(r'\b' + re.escape(to) + r'\b', w) or
                        re.search(r'\b' + re.escape(w) + r'\b', to)
                        for w in scenario_all_words
                    )
                    if not frm_match or not to_match:
                        errors.append(ValidationError(
                            error_type  = "UNVERIFIED_RELATIONSHIP",
                            description = f"Relationship '{frm}' → '{to}' could not be verified from the scenario.",
                            suggestion  = "Double-check whether this relationship is described in the scenario.",
                            severity    = ValidationError.SEVERITY_INFO,
                            element     = f"{frm} → {to}",
                        ))
        # ═════════════════════════════════════════════════════════════════════
        #  NEW: R_REL — Wrong relationship type (scenario expected a different one)
        #  e.g. scenario says "composition" but user drew "association"
        # ═════════════════════════════════════════════════════════════════════
        if scenario_rels and relationships:
            # Build a lookup: (normalized_from, normalized_to) → expected_type
            expected_rel_map: Dict[tuple, str] = {}
            expected_mult_map: Dict[tuple, tuple] = {}  # (frm,to) → (mult_from, mult_to)
            for sr in scenario_rels:
                sf = _normalize(str(sr.get("from", sr.get("source", ""))))
                st = _normalize(str(sr.get("to",   sr.get("target", ""))))
                stype = str(sr.get("type", sr.get("relation", ""))).lower().strip()
                smf   = str(sr.get("multiplicity_from", sr.get("source_multiplicity", ""))).strip()
                smt   = str(sr.get("multiplicity_to",   sr.get("target_multiplicity", ""))).strip()
                if sf and st:
                    expected_rel_map[(sf, st)] = stype
                    expected_mult_map[(sf, st)] = (smf, smt)
            REL_TYPE_MAP = {
                "association": ["association", "line"],
                "generalization": ["generalization", "inheritance", "extends", "inherit"],
                "aggregation": ["aggregation"],
                "composition": ["composition"],
                "dependency": ["dependency", "dashed", "uses"],
            }
            for rel in relationships:
                frm   = _normalize(rel.get("from", ""))
                to    = _normalize(rel.get("to", ""))
                rtype = rel.get("type", "").lower()
                drawn_mult_f = rel.get("multiplicity_from", "").strip()
                drawn_mult_t = rel.get("multiplicity_to",   "").strip()
                if not frm or not to:
                    continue
                # Skip self-loops — S2 (self-association) already handles them
                if frm == to:
                    continue
                # Try to find matching scenario relationship (both directions)
                key     = (frm, to)
                rev_key = (to, frm)
                matched_key = key if key in expected_rel_map else (rev_key if rev_key in expected_rel_map else None)
                if matched_key:
                    exp_type = expected_rel_map[matched_key]
                    exp_mf, exp_mt = expected_mult_map.get(matched_key, ("", ""))
                    # --- Wrong relationship type ---
                    if exp_type:
                        allowed = REL_TYPE_MAP.get(exp_type, [exp_type])
                        # If rtype is empty or a generic connector, don't flag wrong type.
                        # An unlabeled line is treated as a valid association.
                        generic_types = {"", "line", "edge", "link", "connect", "arrow", "connector"}
                        rtype_is_generic = rtype in generic_types
                        # NO_MULTIPLICITY_TYPES: if scenario expects one of these,
                        # a generic/unlabeled line is still the WRONG type — flag it
                        # and skip the multiplicity check (generalization never has multiplicity).
                        exp_is_no_mult_type = any(
                            k in exp_type for k in NO_MULTIPLICITY_TYPES
                        ) if exp_type else False

                        if rtype_is_generic and exp_is_no_mult_type:
                            # User drew a plain line but scenario expects generalization/dependency etc.
                            errors.append(ValidationError(
                                error_type  = "WRONG_RELATIONSHIP_TYPE",
                                description = (
                                    f"Wrong relationship: '{frm}' → '{to}' should be a "
                                    f"'{exp_type}' (inheritance arrow), not a plain association line. "
                                    f"Generalization does not require multiplicity."
                                ),
                                suggestion  = (
                                    f"Delete the association line and draw a '{exp_type}' "
                                    f"(hollow arrowhead) from '{frm}' to '{to}'."
                                ),
                                severity    = ValidationError.SEVERITY_ERROR,
                                element     = f"{frm} → {to}",
                            ))
                            # Skip multiplicity checks — generalization never has multiplicity
                            continue
                        elif rtype_is_generic and exp_type in ("aggregation", "composition"):
                            # User drew a plain line but scenario expects aggregation/composition
                            errors.append(ValidationError(
                                error_type  = "WRONG_RELATIONSHIP_TYPE",
                                description = (
                                    f"Wrong relationship: '{frm}' → '{to}' is drawn as a plain "
                                    f"line, but the scenario expects a '{exp_type}' relationship. "
                                    f"Use the correct arrowhead: {'diamond (open)' if exp_type == 'aggregation' else 'diamond (filled)'}."
                                ),
                                suggestion  = (
                                    f"Delete the plain line and draw a '{exp_type}' arrow "
                                    f"from '{frm}' to '{to}' with multiplicity on both ends."
                                ),
                                severity    = ValidationError.SEVERITY_ERROR,
                                element     = f"{frm} → {to}",
                            ))
                        elif not rtype_is_generic and not any(a in rtype for a in allowed):
                            errors.append(ValidationError(
                                error_type  = "WRONG_RELATIONSHIP_TYPE",
                                description = (
                                    f"Relationship '{frm}' → '{to}' is drawn as '{rtype}', "
                                    f"but the scenario expects a '{exp_type}' relationship."
                                ),
                                suggestion  = f"Change the relationship type to '{exp_type}'.",
                                severity    = ValidationError.SEVERITY_ERROR,
                                element     = f"{frm} → {to}",
                            ))
                    # --- Wrong multiplicity ---
                    if exp_mf and drawn_mult_f and drawn_mult_f != exp_mf:
                        errors.append(ValidationError(
                            error_type  = "WRONG_MULTIPLICITY",
                            description = (
                                f"Multiplicity on '{frm}' side of '{frm}' → '{to}' is '{drawn_mult_f}', "
                                f"but scenario expects '{exp_mf}'."
                            ),
                            suggestion  = f"Change the multiplicity on the '{frm}' side to '{exp_mf}'.",
                            severity    = ValidationError.SEVERITY_ERROR,
                            element     = f"{frm} → {to}",
                        ))
                    if exp_mt and drawn_mult_t and drawn_mult_t != exp_mt:
                        errors.append(ValidationError(
                            error_type  = "WRONG_MULTIPLICITY",
                            description = (
                                f"Multiplicity on '{to}' side of '{frm}' → '{to}' is '{drawn_mult_t}', "
                                f"but scenario expects '{exp_mt}'."
                            ),
                            suggestion  = f"Change the multiplicity on the '{to}' side to '{exp_mt}'.",
                            severity    = ValidationError.SEVERITY_ERROR,
                            element     = f"{frm} → {to}",
                        ))
                else:
                    # Relationship exists in diagram but not found in scenario at all
                    if scenario_rels:
                        # ── Suppress false positive for generalization / inheritance ──
                        # If the drawn relationship is a generalization (inheritance)
                        # and BOTH endpoints are known scenario classes, the inheritance
                        # is implicitly valid — the scenario described the parent class
                        # and subclass(es) even if it didn't list an explicit relationship
                        # entry for every child→parent pair.
                        # Example: scenario mentions Account, SavingsAccount, CurrentAccount
                        # with inheritance keywords → SavingsAccount→Account and
                        # CurrentAccount→Account are valid even if NLP didn't emit them
                        # as explicit scenario_rels entries.
                        _GENERALIZATION_TYPES = {
                            "generalization", "inheritance", "extends", "inherit",
                            "generalisation",
                        }
                        _drawn_rtype = rel.get("type", "").lower().strip()
                        _is_gen_rel  = any(k in _drawn_rtype for k in _GENERALIZATION_TYPES)

                        # Both endpoints must be real scenario classes
                        _sc_norm = {c.lower().replace(" ", "").replace("_", "")
                                    for c in real_scenario_classes}
                        _frm_is_sc = frm in _sc_norm or any(
                            difflib.SequenceMatcher(None, frm, s).ratio() >= 0.82
                            for s in _sc_norm
                        )
                        _to_is_sc  = to in _sc_norm or any(
                            difflib.SequenceMatcher(None, to, s).ratio() >= 0.82
                            for s in _sc_norm
                        )

                        # Also suppress when _global_expected_rel_type is generalization
                        # (scenario text contained inheritance keywords globally) and
                        # both classes appear in the scenario.
                        _global_is_gen = _global_expected_rel_type == "generalization"

                        if _is_gen_rel and _frm_is_sc and _to_is_sc:
                            pass  # Valid inheritance between scenario classes — no error
                        elif _global_is_gen and _frm_is_sc and _to_is_sc and not _drawn_rtype:
                            pass  # Unlabeled line between scenario classes when global type is generalization
                        else:
                            # Bug-2 fix: suppress EXTRA_RELATIONSHIP when both
                            # endpoints are known scenario classes — the user drew
                            # a logically valid implicit relationship (e.g. a
                            # customer existing inside a bank system).  Flagging it
                            # confuses students who correctly modelled the domain.
                            _sc_norm_extra = {
                                c.lower().replace(" ","").replace("_","")
                                for c in real_scenario_classes
                            }
                            def _in_sc(name):
                                n = name.lower().replace(" ","").replace("_","")
                                if n in _sc_norm_extra:
                                    return True
                                return any(
                                    difflib.SequenceMatcher(None, n, s).ratio() >= 0.82
                                    for s in _sc_norm_extra
                                )
                            if _in_sc(frm) and _in_sc(to):
                                pass  # Both endpoints are scenario classes — implicit rel is valid
                            else:
                                errors.append(ValidationError(
                                    error_type  = "EXTRA_RELATIONSHIP",
                                    description = (
                                        f"Relationship '{frm}' → '{to}' is not described in the scenario."
                                    ),
                                    suggestion  = "Remove this relationship or verify it is required by the scenario.",
                                    severity    = ValidationError.SEVERITY_WARNING,
                                    element     = f"{frm} → {to}",
                                ))
        # ═════════════════════════════════════════════════════════════════════
        #  NEW: R_MISSING_REL — Relationships in scenario not drawn at all
        #
        #  FIX (Bug 2 & 3):
        #  Previously this block only ran when scenario_rels was non-empty.
        #  But generalization (inheritance) pairs are often detected via
        #  _global_expected_rel_type rather than per-pair scenario_rels entries.
        #  Similarly, aggregation/composition pairs may come ONLY from the
        #  classifier.  If the user deletes such relationships, no error fired.
        #
        #  Fix: always run this block when scenario_has_relationships is True,
        #  even if scenario_rels is empty.  When scenario_rels IS empty but
        #  _global_expected_rel_type is "generalization"/"aggregation"/etc.,
        #  synthesize expected pairs from real_scenario_classes using the
        #  classifier results or the IS-A heuristic output.
        # ═════════════════════════════════════════════════════════════════════

        # ── Helper: build ID→label lookup ─────────────────────────────────────
        id_to_label: Dict[str, str] = {}
        for s in shapes:
            sid = str(s.get("id", s.get("shapeId", s.get("nodeId", "")))).strip()
            lbl = _get_class_label(s)
            if sid and lbl:
                id_to_label[sid.lower()] = lbl

        def _resolve_label(raw: str) -> str:
            """Convert node ID to class label if possible, else return raw."""
            r = raw.strip()
            resolved = id_to_label.get(r.lower(), "")
            if resolved:
                return resolved
            matched = self._fuzzy_match(r, class_labels)
            if matched:
                return r
            return r

        # ── Build drawn pairs using resolved labels ────────────────────────────
        drawn_rel_list: List[tuple] = []
        for rel in relationships:
            frm_raw = rel.get("from", "")
            to_raw  = rel.get("to",   "")
            frm_resolved = _normalize(_resolve_label(frm_raw))
            to_resolved  = _normalize(_resolve_label(to_raw))
            if frm_resolved and to_resolved:
                drawn_rel_list.append((frm_resolved, to_resolved))

        def _drawn_pair_exists(sf: str, st: str) -> bool:
            """Check if a drawn relationship exists between sf and st (fuzzy, bidirectional)."""
            for (df, dt) in drawn_rel_list:
                forward  = (self._fuzzy_match(sf, [df]) and self._fuzzy_match(st, [dt]))
                backward = (self._fuzzy_match(sf, [dt]) and self._fuzzy_match(st, [df]))
                if forward or backward:
                    return True
            # Also check raw (unresolved) relationships
            for rel in relationships:
                frm_raw = _normalize(rel.get("from", ""))
                to_raw  = _normalize(rel.get("to",   ""))
                if not frm_raw or not to_raw:
                    continue
                forward  = (self._fuzzy_match(sf, [frm_raw]) and self._fuzzy_match(st, [to_raw]))
                backward = (self._fuzzy_match(sf, [to_raw])  and self._fuzzy_match(st, [frm_raw]))
                if forward or backward:
                    return True
            return False

        def _drawn_pair_exists_with_type(sf: str, st: str, expected_types: tuple) -> bool:
            """
            Check if a drawn relationship exists between sf and st AND is of the correct type.
            For generalization: user must have drawn a generalization arrow specifically.
            For aggregation/composition: must match the expected type.
            If expected_types is empty, any relationship type counts.
            """
            for rel in relationships:
                frm_raw = _normalize(_resolve_label(rel.get("from", "")))
                to_raw  = _normalize(_resolve_label(rel.get("to",   "")))
                rtype   = rel.get("type", "").lower().strip()
                if not frm_raw or not to_raw:
                    frm_raw = _normalize(rel.get("from", ""))
                    to_raw  = _normalize(rel.get("to",   ""))
                forward  = (self._fuzzy_match(sf, [frm_raw]) and self._fuzzy_match(st, [to_raw]))
                backward = (self._fuzzy_match(sf, [to_raw])  and self._fuzzy_match(st, [frm_raw]))
                if forward or backward:
                    if not expected_types:
                        return True  # any type matches
                    if any(k in rtype for k in expected_types):
                        return True
            return False

        # ── Check per-pair scenario_rels (existing behaviour) ─────────────────
        if scenario_rels:
            for sr in scenario_rels:
                sf    = _normalize(str(sr.get("from", sr.get("source", ""))))
                st    = _normalize(str(sr.get("to",   sr.get("target", ""))))
                stype = str(sr.get("type", sr.get("relation", "association"))).lower().strip()
                if not sf or not st:
                    continue

                if not _drawn_pair_exists(sf, st):
                    # Bug-3 fix: clear actionable message for any missing relationship type
                    _missing_stype = stype if stype else "association"
                    _is_gen_missing = any(k in _missing_stype for k in
                        ("generalization", "inheritance", "extends", "inherit"))
                    errors.append(ValidationError(
                        error_type  = "MISSING_RELATIONSHIP",
                        description = (
                            f"Missing relationship between '{sf}' and '{st}'. "
                            f"The scenario requires a '{_missing_stype}' relationship here, "
                            f"but it is not drawn in the diagram."
                        ),
                        suggestion  = (
                            f"Add a generalization relationship: draw a hollow triangle arrowhead "
                            f"from '{sf}' (child) to '{st}' (parent). "
                            f"Do NOT add multiplicity on generalization arrows."
                            if _is_gen_missing else
                            f"Add a '{_missing_stype}' relationship from '{sf}' to '{st}'."
                        ),
                        severity    = ValidationError.SEVERITY_ERROR,
                        element     = f"{sf} → {st}",
                    ))

        # ── FIX Bug 2: Missing generalization check ────────────────────────────
        # When scenario implies generalization (via keywords or classifier) but
        # the user has removed the inheritance arrows, we must report the error.
        # This runs REGARDLESS of whether scenario_rels already covered this pair.
        if _global_expected_rel_type == "generalization" and real_scenario_classes:
            _GEN_DRAWN_TYPES = ("generalization", "inheritance", "extends", "inherit",
                                "generalisation")
            # Find parent-child pairs from classifier results
            _gen_pairs_to_check: List[tuple] = []
            for _cr in _classifier_rels:
                if _cr.get("type") == "generalization":
                    _gen_pairs_to_check.append(
                        (_normalize(_cr["from"]), _normalize(_cr["to"]))
                    )
            # If classifier found no pairs, synthesise from IS-A heuristics
            if not _gen_pairs_to_check:
                _sc_lower_list = [c.lower() for c in real_scenario_classes]
                for _parent, _children in {
                    "staff":    {"doctor", "nurse", "admin"},
                    "person":   {"student", "teacher", "employee", "manager", "doctor",
                                 "nurse", "admin", "administrator", "customer",
                                 "professor", "instructor", "worker"},
                    "account":  {"savingsaccount", "currentaccount", "checkingaccount",
                                 "loanaccount", "bankaccount", "creditaccount"},
                    "employee": {"manager", "developer", "engineer", "designer",
                                 "intern", "director", "supervisor"},
                    "animal":   {"duck", "fish", "zebra", "dog", "cat", "bird",
                                 "lion", "tiger", "horse", "snake", "whale"},
                    "vehicle":  {"car", "truck", "bus", "bike", "motorcycle",
                                 "bicycle", "train", "plane", "boat", "ship"},
                    "shape":    {"circle", "square", "rectangle", "triangle",
                                 "polygon", "oval", "ellipse"},
                }.items():
                    if _parent in _sc_lower_list:
                        for _child in _children:
                            if _child in _sc_lower_list:
                                _gen_pairs_to_check.append((_child, _parent))

            # Also check scenario_rels for generalization pairs not yet caught
            _already_checked: Set[tuple] = set()
            for _cr in scenario_rels:
                _ctype = str(_cr.get("type", "")).lower()
                if any(k in _ctype for k in _GEN_DRAWN_TYPES):
                    _cf = _normalize(str(_cr.get("from", _cr.get("source", ""))))
                    _ct = _normalize(str(_cr.get("to",   _cr.get("target", ""))))
                    if _cf and _ct:
                        _gen_pairs_to_check.append((_cf, _ct))

            _reported_gen: Set[tuple] = set()
            for (_gf, _gt) in _gen_pairs_to_check:
                _canon = tuple(sorted([_gf, _gt]))
                if _canon in _reported_gen:
                    continue
                _reported_gen.add(_canon)
                # Check if a GENERALIZATION arrow (specifically) is drawn
                if not _drawn_pair_exists_with_type(_gf, _gt, _GEN_DRAWN_TYPES):
                    # Only report if both classes exist in the diagram
                    _gf_in_diagram = self._fuzzy_match(_gf, [l.lower() for l in class_labels])
                    _gt_in_diagram = self._fuzzy_match(_gt, [l.lower() for l in class_labels])
                    if _gf_in_diagram and _gt_in_diagram:
                        # Check not already reported by per-pair loop above
                        _already_in_errors = any(
                            e.error_type == "MISSING_RELATIONSHIP"
                            and _gf in e.element and _gt in e.element
                            for e in errors
                        )
                        if not _already_in_errors:
                            errors.append(ValidationError(
                                error_type  = "MISSING_RELATIONSHIP",
                                description = (
                                    f"Missing relationship between '{_gf}' and '{_gt}'. "
                                    f"The scenario requires a generalization (inheritance) "
                                    f"arrow here, but none is drawn."
                                ),
                                suggestion  = (
                                    f"Add a generalization relationship: draw a hollow triangle arrowhead "
                                    f"from '{_gf}' (child class) pointing to '{_gt}' (parent class). "
                                    f"Do NOT add multiplicity on generalization arrows."
                                ),
                                severity    = ValidationError.SEVERITY_ERROR,
                                element     = f"{_gf} → {_gt}",
                            ))

        # ── FIX Bug 3: Missing aggregation / composition check ─────────────────
        # When scenario expects aggregation (weak) or composition (strong) and
        # the user deletes that arrow, report it clearly.
        if _global_expected_rel_type in ("aggregation", "composition") and real_scenario_classes:
            _TYPE_KEYWORDS = {
                "aggregation": ("aggregation",),
                "composition": ("composition",),
            }
            _drawn_type_keywords = _TYPE_KEYWORDS.get(_global_expected_rel_type, ())
            # Find pairs from classifier
            _hasa_pairs: List[tuple] = []
            for _cr in _classifier_rels:
                if _cr.get("type") == _global_expected_rel_type:
                    _hasa_pairs.append(
                        (_normalize(_cr["from"]), _normalize(_cr["to"]))
                    )
            # Also from scenario_rels
            for _cr in scenario_rels:
                if _global_expected_rel_type in str(_cr.get("type", "")).lower():
                    _cf = _normalize(str(_cr.get("from", _cr.get("source", ""))))
                    _ct = _normalize(str(_cr.get("to",   _cr.get("target", ""))))
                    if _cf and _ct:
                        _hasa_pairs.append((_cf, _ct))

            _reported_hasa: Set[tuple] = set()
            for (_hf, _ht) in _hasa_pairs:
                _canon = tuple(sorted([_hf, _ht]))
                if _canon in _reported_hasa:
                    continue
                _reported_hasa.add(_canon)
                if not _drawn_pair_exists_with_type(_hf, _ht, _drawn_type_keywords):
                    _hf_in_diagram = self._fuzzy_match(_hf, [l.lower() for l in class_labels])
                    _ht_in_diagram = self._fuzzy_match(_ht, [l.lower() for l in class_labels])
                    if _hf_in_diagram and _ht_in_diagram:
                        _already_in_errors = any(
                            e.error_type == "MISSING_RELATIONSHIP"
                            and _hf in e.element and _ht in e.element
                            for e in errors
                        )
                        if not _already_in_errors:
                            _rel_label = (
                                "aggregation (open diamond — weak/shared part-of)"
                                if _global_expected_rel_type == "aggregation"
                                else "composition (filled diamond — strong/exclusive part-of)"
                            )
                            errors.append(ValidationError(
                                error_type  = "MISSING_RELATIONSHIP",
                                description = (
                                    f"Missing relationship between '{_hf}' and '{_ht}'. "
                                    f"The scenario requires a '{_global_expected_rel_type}' "
                                    f"relationship here, but it is not drawn in the diagram."
                                ),
                                suggestion  = (
                                    f"Add a {_rel_label} arrow "
                                    f"from '{_hf}' to '{_ht}' with multiplicity on both ends."
                                ),
                                severity    = ValidationError.SEVERITY_ERROR,
                                element     = f"{_hf} → {_ht}",
                            ))
        # ═════════════════════════════════════════════════════════════════════
        #  R_ASSOC_LABEL — Association label validation
        #  Rule (a): Scenario mentions a label for a relationship pair
        #            AND the user DID draw that relationship pair
        #            BUT did NOT put the label on it → ERROR.
        #  Rule (b): Scenario does NOT mention a label for a pair →
        #            no error, whether the user drew a label or not.
        #  NOTE: If the user did not draw the pair at all, MISSING_RELATIONSHIP
        #        (above) already handles that — we do NOT also report a missing
        #        label for an undrawn relationship.
        # ═════════════════════════════════════════════════════════════════════
        if scenario_rels and relationships:
            # ── Build (frm_n, to_n) → expected_label from scenario_rels ──────
            # Only store pairs where scenario explicitly provides a label.
            _expected_label_map: Dict[tuple, str] = {}
            for _sr in scenario_rels:
                _sf = _normalize(str(_sr.get("from", _sr.get("source", ""))))
                _st = _normalize(str(_sr.get("to",   _sr.get("target", ""))))
                if not _sf or not _st:
                    continue
                # Self-loop relationships don't get label requirements
                if _sf == _st:
                    continue
                # NLP extractor may supply label under several field names
                _exp_lbl = ""
                for _lf in ("label","name","role","association_name","relation_name","assoc_label"):
                    _v = str(_sr.get(_lf, "")).strip()
                    if _v and _v.lower() not in ("none","null","","undefined"):
                        _exp_lbl = _v
                        break
                # If no explicit label field, try extracting the relationship verb
                # from the trigger phrase stored by the classifier.
                # e.g. trigger="Bank manages Customer" → label="manages"
                # This covers scenarios where the label is the verb in the sentence
                # ("Bank manages Customer", "Customer has Account", etc.)
                #
                # IMPORTANT: NLP-extractor entries that were already in scenario_rels
                # before the classifier ran do NOT have a "trigger" field.  For those
                # entries we fall back to scanning the raw scenario text directly.
                # Known association/aggregation verbs that serve as labels
                _LABEL_VERBS = (
                    "manages", "holds",
                    "owns", "provides", "maintains", "employs",
                    "handles", "processes", "serves", "assigns",
                    "registers", "tracks", "records", "stores",
                    "issues", "generates", "creates", "enrolls",
                    "supervises", "coordinates", "schedules",
                )
                # Also check multi-word verbs: "has a", "is a", "is an", "is associated with"
                _LABEL_PHRASES = (
                    "is associated with", "is related to",
                    "is part of", "belongs to",
                )
                if not _exp_lbl:
                    _trigger = str(_sr.get("trigger", "")).strip()
                    # IMPORTANT: only scan the trigger phrase (from the classifier).
                    # Do NOT fall back to the full raw scenario text — that causes
                    # false positives by finding verbs from completely unrelated sentences.
                    # NLP-extractor entries (no trigger) never auto-get a label this way.
                    if _trigger:
                        for _lp in _LABEL_PHRASES:
                            if _lp in _trigger.lower():
                                _exp_lbl = _lp.split()[0]  # first word
                                break
                        if not _exp_lbl:
                            for _lv in _LABEL_VERBS:
                                if f" {_lv} " in f" {_trigger.lower()} ":
                                    _exp_lbl = _lv
                                    break
                # Only relevant for association / aggregation / composition
                # NOTE: classifier-added entries often have type="" or "association"
                # so we accept empty type as well (do NOT restrict to only mult types).
                _sr_type = str(_sr.get("type", _sr.get("relation",""))).lower()
                _is_no_mult_rel = any(k in _sr_type for k in
                    ("generalization","inheritance","extends","inherit",
                     "realization","dependency","depend"))
                # Rule (b): if scenario has NO label for this pair, do not store it
                # (labels are entirely optional when scenario doesn't mention one).
                # We store label for association/aggregation/composition AND for
                # empty-type entries (classifier-produced) as long as label exists.
                if not _is_no_mult_rel and _exp_lbl:
                    # Store canonical pair (both directions) to simplify lookup
                    _key_fwd = (_sf, _st)
                    _key_rev = (_st, _sf)
                    if _key_fwd not in _expected_label_map:
                        _expected_label_map[_key_fwd] = _exp_lbl
                    if _key_rev not in _expected_label_map:
                        _expected_label_map[_key_rev] = _exp_lbl

            # ── Build set of class pairs the user actually drew ───────────────
            # Only consider a label error if the relationship was drawn.
            _drawn_pairs_norm: set = set()
            _drawn_label_map: Dict[tuple, str] = {}
            for _rel in relationships:
                _rf = _normalize(_rel.get("from", ""))
                _rt = _normalize(_rel.get("to",   ""))
                _rl = str(_rel.get("label", "")).strip()
                if _rf and _rt:
                    _drawn_pairs_norm.add((_rf, _rt))
                    _drawn_pairs_norm.add((_rt, _rf))  # bidirectional
                    _drawn_label_map[(_rf, _rt)] = _rl
                    _drawn_label_map[(_rt, _rf)] = _rl

            # ── Rule (a): scenario has a label AND pair was drawn → check label ─
            _reported_lbl_pairs: set = set()
            for (_sf, _st), _exp_lbl in _expected_label_map.items():
                # Deduplicate: report each unordered pair only once
                _canonical = tuple(sorted([_sf, _st]))
                if _canonical in _reported_lbl_pairs:
                    continue

                # *** KEY GUARD: only check label if the user actually drew this pair ***
                # If the pair was not drawn at all, MISSING_RELATIONSHIP already reported it.
                _pair_was_drawn = (
                    (_sf, _st) in _drawn_pairs_norm or
                    (_st, _sf) in _drawn_pairs_norm
                )
                if not _pair_was_drawn:
                    # Don't double-report: MISSING_RELATIONSHIP covers this case.
                    _reported_lbl_pairs.add(_canonical)
                    continue

                _drawn_lbl = _drawn_label_map.get((_sf, _st), "")
                # Fuzzy match: drawn label ≥ 75% similar → accept
                if _drawn_lbl:
                    _sim = difflib.SequenceMatcher(
                        None, _drawn_lbl.lower(), _exp_lbl.lower()
                    ).ratio()
                    if _sim >= 0.75:
                        _reported_lbl_pairs.add(_canonical)
                        continue  # label is present and close enough
                # Label missing or too different → report error
                _reported_lbl_pairs.add(_canonical)
                errors.append(ValidationError(
                    error_type  = "MISSING_ASSOCIATION_LABEL",
                    description = (
                        f"The association between '{_sf}' and '{_st}' "
                        f"should have the label '{_exp_lbl}' as described "
                        f"in the scenario, but no label is drawn on the relationship."
                    ),
                    suggestion  = (
                        f"Add the label '{_exp_lbl}' to the relationship "
                        f"between '{_sf}' and '{_st}'."
                    ),
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = f"{_sf} → {_st}",
                ))
            # Rule (b): no label in scenario → no error (labels are optional).
            # We never flag a relationship just because it HAS a label that
            # wasn't mentioned in the scenario — users are free to add labels.

        # ═════════════════════════════════════════════════════════════════════
        #  R_GLOBAL_WRONG_TYPE — Fallback wrong-type check via global keyword
        # ─────────────────────────────────────────────────────────────────────
        #  WHY THIS EXISTS:
        #  R_REL (above) only fires when scenario_rels has per-pair entries from
        #  the NLP extractor.  In two common cases that never happens:
        #
        #  Case A — Explicit keyword, no per-pair data:
        #    Scenario: "draw a class diagram with generalization relationship
        #               between Animal, Duck, Fish, Zebra"
        #    → NLP may emit scenario_rels=[] but the word "generalization" is
        #      in the raw text.  _global_expected_rel_type = "generalization".
        #
        #  Case B — No keyword at all (implicit IS-A):
        #    Scenario: "draw a class diagram of Animal having Duck Fish Zebra"
        #    → IS-A heuristic detected Animal as parent → duck/fish/zebra as
        #      children.  _global_expected_rel_type = "generalization".
        #
        #  In both cases if the student draws "association" the validator
        #  previously said "Diagram is Correct" — this block fixes that.
        # ═════════════════════════════════════════════════════════════════════
        if (not scenario_rels) and _global_expected_rel_type and relationships:
            _GEN_FAMILY   = {"generalization","inheritance","extends","inherit"}
            _ASSOC_FAMILY = {"association","line","edge","link",
                             "connect","arrow","connector",""}
            _AGG_FAMILY   = {"aggregation"}
            _COMP_FAMILY  = {"composition"}

            _exp_is_gen   = _global_expected_rel_type == "generalization"
            _exp_is_assoc = _global_expected_rel_type == "association"
            _exp_is_agg   = _global_expected_rel_type == "aggregation"
            _exp_is_comp  = _global_expected_rel_type == "composition"

            for _grel in relationships:
                _grtype = _grel.get("type", "").lower().strip()
                _grf    = _normalize(_grel.get("from", ""))
                _grt    = _normalize(_grel.get("to",   ""))
                if not _grf or not _grt:
                    continue

                _drawn_is_gen   = any(k in _grtype for k in _GEN_FAMILY)
                _drawn_is_assoc = _grtype in _ASSOC_FAMILY or "association" in _grtype
                _drawn_is_agg   = "aggregation" in _grtype
                _drawn_is_comp  = "composition"  in _grtype

                _gtype_wrong = False
                if _exp_is_gen and not _drawn_is_gen:
                    _gtype_wrong = True
                elif _exp_is_assoc and (_drawn_is_gen or _drawn_is_agg or _drawn_is_comp):
                    _gtype_wrong = True
                elif _exp_is_agg and not _drawn_is_agg:
                    _gtype_wrong = True
                elif _exp_is_comp and not _drawn_is_comp:
                    _gtype_wrong = True

                if _gtype_wrong:
                    _drawn_lbl = _grtype if _grtype else "association (plain line)"
                    # Bug-5 fix: tailor the description and suggestion based on
                    # what the scenario ACTUALLY expects, not just what was drawn.
                    # Specifically: if user drew generalization but scenario expects
                    # association/aggregation/composition, tell them to use the correct
                    # HAS-A arrow — do NOT say "draw generalization".
                    _drawn_is_gen_wrong = any(k in _grtype for k in
                        ("generalization", "inheritance", "extends", "inherit"))
                    if _exp_is_assoc or _exp_is_agg or _exp_is_comp:
                        _arrow_hint = {
                            "association": "a solid line with an open arrowhead (association)",
                            "aggregation": "an open diamond arrowhead (aggregation — weak part-of)",
                            "composition": "a filled diamond arrowhead (composition — strong part-of)",
                        }.get(_global_expected_rel_type, f"a '{_global_expected_rel_type}' arrow")
                        _desc_extra = (
                            " Generalization (inheritance) is only for IS-A relationships. "
                            f"'{_grf}' does not inherit '{_grt}' — it uses or contains it."
                            if _drawn_is_gen_wrong else ""
                        )
                        _sugg_extra = (
                            f" Replace the generalization arrow with {_arrow_hint} "
                            f"and add multiplicity labels on both ends."
                            if _drawn_is_gen_wrong else
                            f" Use {_arrow_hint} and add multiplicity labels on both ends."
                        )
                    elif _exp_is_gen:
                        _desc_extra = (
                            " Generalization uses a hollow triangle arrowhead pointing "
                            "from the child class UP to the parent class. "
                            "No multiplicity labels are needed on inheritance arrows."
                        )
                        _sugg_extra = (
                            " Use the generalization/inheritance tool (hollow arrowhead). "
                            "Remove any multiplicity labels from the arrow."
                        )
                    else:
                        _desc_extra = ""
                        _sugg_extra = ""
                    errors.append(ValidationError(
                        error_type  = "WRONG_RELATIONSHIP_TYPE",
                        description = (
                            f"Wrong relationship type: '{_grf}' → '{_grt}' is drawn as "
                            f"'{_drawn_lbl}', but the scenario requires a "
                            f"'{_global_expected_rel_type}' relationship."
                            + _desc_extra
                        ),
                        suggestion  = (
                            f"Delete the '{_drawn_lbl}' and draw a "
                            f"'{_global_expected_rel_type}' relationship from '{_grf}' to '{_grt}'."
                            + _sugg_extra
                        ),
                        severity    = ValidationError.SEVERITY_ERROR,
                        element     = f"{_grf} → {_grt}",
                    ))

        # ═════════════════════════════════════════════════════════════════════
        #  R7 — Duplicate class names
        # ═════════════════════════════════════════════════════════════════════
        seen: Set[str] = set()
        for cls in class_labels:
            cls_lower = _normalize(cls)
            if cls_lower in seen:
                errors.append(ValidationError(
                    error_type  = "DUPLICATE_CLASS",
                    description = f"Class '{cls}' appears more than once in the diagram.",
                    suggestion  = f"Remove the duplicate '{cls}'. Each class must appear only once.",
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = cls,
                ))
            seen.add(cls_lower)
        # ═════════════════════════════════════════════════════════════════════
        #  R16 — Isolated classes
        #  Fire whenever the diagram has at least one drawn relationship but
        #  a class is not part of any of them — user clearly knows how to
        #  draw relationships but forgot to connect this class.
        # ═════════════════════════════════════════════════════════════════════
        # R16: Only fire if there ARE valid resolved relationships in the diagram
        # (after filtering out '?' endpoints above). If relationships list is empty
        # it means the app hasn't stored endpoints in a way we can read — don't
        # falsely flag every class as isolated.
        valid_rels_for_isolated = [
            r for r in relationships
            if r.get("from","").strip() not in ("","?")
            and r.get("to","").strip()   not in ("","?")
        ]
        if valid_rels_for_isolated:
            connected: Set[str] = set()
            for rel in valid_rels_for_isolated:
                connected.add(_normalize(rel.get("from", "")))
                connected.add(_normalize(rel.get("to",   "")))
            for cls in class_labels:
                cls_norm = _normalize(cls)
                # Use fuzzy match: check if this class appears in any relationship endpoint
                if cls_norm not in connected and not self._fuzzy_match(cls, list(connected)):
                    errors.append(ValidationError(
                        error_type  = "ISOLATED_CLASS",
                        description = f"Class '{cls}' is not connected to any other class in the diagram.",
                        suggestion  = f"Draw a relationship from '{cls}' to at least one other class.",
                        severity    = ValidationError.SEVERITY_ERROR,
                        element     = cls,
                    ))
        # ═════════════════════════════════════════════════════════════════════
        # =============================================================================
        #  SEMANTIC ANALYSIS
        #  S1.  Inheritance direction check  - child should point TO parent
        #  S2.  Self-association detection   - class related to itself
        #  S3.  IS-A vs HAS-A mismatch       - wrong relationship type semantics
        #  S4.  Invalid multiplicity values  - e.g. 0..0 is meaningless
        #  S5.  Multiplicity on generalization - inheritance never has mult labels
        #  S6.  Redundant generalization     - A->C when A->B->C already exists
        #  S7.  Circular inheritance         - A inherits B AND B inherits A
        #  S8.  Abstract class heuristic     - class with 3+ children, no attrs
        # =============================================================================

        def _s_is_generalization(rel):
            t = rel.get("type", "").lower()
            return any(k in t for k in ("generalization","inheritance","extends","inherit"))

        def _s_is_association(rel):
            t = rel.get("type", "").lower().strip()
            return t in ("association","line","edge","link","connect","arrow","connector","") \
                   or "association" in t

        def _s_is_aggregation(rel):
            return "aggregation" in rel.get("type","").lower()

        def _s_is_composition(rel):
            return "composition" in rel.get("type","").lower()

        # Resolve from/to labels using id_to_label_mult (built in R15b section)
        def _s_res(raw):
            r = raw.strip()
            return id_to_label_mult.get(r.lower(), r)

        sem_rels = []
        for _r in relationships:
            _f = _s_res(_r.get("from",""))
            _t = _s_res(_r.get("to",""))
            if _f and _t:
                sem_rels.append({
                    "from": _f, "to": _t,
                    "type": _r.get("type","").lower().strip(),
                    "mf":   _r.get("multiplicity_from","").strip(),
                    "mt":   _r.get("multiplicity_to","").strip(),
                })

        # Build generalization edges set  (child, parent)
        _gen_edges = set()
        for _r in sem_rels:
            if _s_is_generalization(_r):
                _gen_edges.add((_normalize(_r["from"]), _normalize(_r["to"])))

        # S1: Inheritance direction check
        if scenario_rels:
            _child_of: Dict[str, Set[str]] = {}
            for _sr in scenario_rels:
                _sf = _normalize(str(_sr.get("from", _sr.get("source",""))))
                _st = _normalize(str(_sr.get("to",   _sr.get("target",""))))
                _st2 = str(_sr.get("type", _sr.get("relation",""))).lower()
                if _sf and _st and any(k in _st2 for k in
                        ("generalization","inheritance","extends","inherit")):
                    _child_of.setdefault(_sf, set()).add(_st)
            for _r in sem_rels:
                if not _s_is_generalization(_r):
                    continue
                _fn = _normalize(_r["from"])
                _tn = _normalize(_r["to"])
                if _tn in _child_of and _fn in _child_of[_tn]:
                    errors.append(ValidationError(
                        error_type  = "WRONG_GENERALIZATION_DIRECTION",
                        description = (
                            f"Inheritance arrow '{_r['from']}' to '{_r['to']}' is "
                            f"drawn in the WRONG direction. '{_r['to']}' is the child "
                            f"class — the arrow should go FROM '{_r['to']}' TO '{_r['from']}'."
                        ),
                        suggestion  = (
                            f"Reverse the arrow: draw it from '{_r['to']}' (child) "
                            f"to '{_r['from']}' (parent)."
                        ),
                        severity    = ValidationError.SEVERITY_ERROR,
                        element     = f"{_r['from']} -> {_r['to']}",
                    ))

        # S2: Self-association
        for _r in sem_rels:
            if _normalize(_r["from"]) == _normalize(_r["to"]):
                errors.append(ValidationError(
                    error_type  = "SEMANTIC_SELF_ASSOCIATION",
                    description = (
                        f"Class '{_r['from']}' has a relationship pointing to itself. "
                        f"Self-associations are almost never correct in basic class diagrams."
                    ),
                    suggestion  = (
                        f"Remove the self-loop on '{_r['from']}'. "
                        f"If this is intentional (e.g. recursive structure), "
                        f"add a comment explaining why."
                    ),
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = _r["from"],
                ))

        # S3: IS-A vs HAS-A mismatch
        if scenario_rels:
            for _sr in scenario_rels:
                _sf   = _normalize(str(_sr.get("from", _sr.get("source",""))))
                _st   = _normalize(str(_sr.get("to",   _sr.get("target",""))))
                _stype = str(_sr.get("type", _sr.get("relation",""))).lower()
                _drawn = [
                    _r for _r in sem_rels
                    if (_normalize(_r["from"]) == _sf and _normalize(_r["to"]) == _st)
                    or (_normalize(_r["from"]) == _st and _normalize(_r["to"]) == _sf)
                ]
                _isa_expected  = any(k in _stype for k in
                    ("generalization","inheritance","extends","inherit"))
                _hasa_expected = any(k in _stype for k in
                    ("aggregation","composition","association"))
                for _dr in _drawn:
                    _drawn_hasa = _s_is_aggregation(_dr) or _s_is_composition(_dr)
                    _drawn_isa  = _s_is_generalization(_dr)
                    if _isa_expected and _drawn_hasa:
                        errors.append(ValidationError(
                            error_type  = "SEMANTIC_ISA_VS_HASA",
                            description = (
                                f"'{_dr['from']}' to '{_dr['to']}' is drawn as "
                                f"'{_dr['type']}' (HAS-A relationship), but the scenario "
                                f"implies an IS-A (inheritance) relationship. "
                                f"'{_dr['from']}' IS a type of '{_dr['to']}', not something it owns."
                            ),
                            suggestion  = (
                                f"Replace '{_dr['type']}' with a generalization "
                                f"(hollow arrowhead) — '{_dr['from']}' IS-A '{_dr['to']}'."
                            ),
                            severity    = ValidationError.SEVERITY_ERROR,
                            element     = f"{_dr['from']} -> {_dr['to']}",
                        ))
                    if _hasa_expected and _drawn_isa:
                        errors.append(ValidationError(
                            error_type  = "SEMANTIC_ISA_VS_HASA",
                            description = (
                                f"'{_dr['from']}' to '{_dr['to']}' is drawn as "
                                f"generalization (IS-A), but the scenario implies "
                                f"a HAS-A ('{_stype}') relationship. "
                                f"'{_dr['from']}' does not inherit '{_dr['to']}' — it uses/contains it."
                            ),
                            suggestion  = (
                                f"Replace the generalization arrow with a '{_stype}' "
                                f"to model that '{_dr['from']}' HAS/USES '{_dr['to']}'."
                            ),
                            severity    = ValidationError.SEVERITY_ERROR,
                            element     = f"{_dr['from']} -> {_dr['to']}",
                        ))

        # S4: Semantically invalid multiplicity (0..0)
        _ZERO_ZERO = re.compile(r"^0\.\.0$")
        for _r in sem_rels:
            for _mv, _side in [(_r["mf"], _r["from"]), (_r["mt"], _r["to"])]:
                if _mv and _ZERO_ZERO.match(_mv):
                    errors.append(ValidationError(
                        error_type  = "SEMANTIC_INVALID_MULTIPLICITY",
                        description = (
                            f"Multiplicity '0..0' on '{_side}' side of "
                            f"'{_r['from']}' to '{_r['to']}' is semantically meaningless. "
                            f"'0..0' means zero objects are ever involved."
                        ),
                        suggestion  = (
                            f"Replace '0..0' with a valid value such as "
                            f"'0..1', '1', '0..*', or '1..*'."
                        ),
                        severity    = ValidationError.SEVERITY_ERROR,
                        element     = f"{_side} (in {_r['from']} -> {_r['to']})",
                    ))

        # S5: Multiplicity on generalization arrow
        for _r in sem_rels:
            if _s_is_generalization(_r) and (_r["mf"] or _r["mt"]):
                errors.append(ValidationError(
                    error_type  = "SEMANTIC_MULTIPLICITY_ON_GENERALIZATION",
                    description = (
                        f"Generalization '{_r['from']}' to '{_r['to']}' has "
                        f"multiplicity labels, which is incorrect in UML. "
                        f"Inheritance arrows never carry multiplicity."
                    ),
                    suggestion  = (
                        f"Remove the multiplicity labels from the "
                        f"'{_r['from']}' to '{_r['to']}' generalization arrow."
                    ),
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = f"{_r['from']} -> {_r['to']}",
                ))

        # S6: Redundant generalization (transitive — A->C when A->B->C exists)
        def _reachable_parents(start, edges):
            visited, queue = set(), [start]
            while queue:
                node = queue.pop()
                for (f2, t2) in edges:
                    if f2 == node and t2 not in visited:
                        visited.add(t2)
                        queue.append(t2)
            return visited

        for (_child, _parent) in list(_gen_edges):
            _remaining = _gen_edges - {(_child, _parent)}
            if _parent in _reachable_parents(_child, _remaining):
                errors.append(ValidationError(
                    error_type  = "SEMANTIC_REDUNDANT_GENERALIZATION",
                    description = (
                        f"Generalization '{_child.title()}' to '{_parent.title()}' is "
                        f"redundant: '{_child.title()}' already inherits "
                        f"'{_parent.title()}' through another class in the hierarchy."
                    ),
                    suggestion  = (
                        f"Remove the direct generalization from "
                        f"'{_child.title()}' to '{_parent.title()}'. "
                        f"It is already implied by the existing inheritance chain."
                    ),
                    severity    = ValidationError.SEVERITY_WARNING,
                    element     = f"{_child} -> {_parent}",
                ))

        # S7: Circular inheritance (A->B and B->A both generalization)
        _reported_circular = set()
        for (_child, _parent) in list(_gen_edges):
            _pair = tuple(sorted([_child, _parent]))
            if (_parent, _child) in _gen_edges and _pair not in _reported_circular:
                _reported_circular.add(_pair)
                errors.append(ValidationError(
                    error_type  = "SEMANTIC_CIRCULAR_INHERITANCE",
                    description = (
                        f"Circular inheritance: '{_child.title()}' inherits "
                        f"'{_parent.title()}' AND '{_parent.title()}' inherits "
                        f"'{_child.title()}'. This is logically impossible in OOP."
                    ),
                    suggestion  = (
                        f"Remove one of the two generalization arrows. "
                        f"Decide which is the parent class and which is the child."
                    ),
                    severity    = ValidationError.SEVERITY_ERROR,
                    element     = f"{_child} <-> {_parent}",
                ))

        # S8: Abstract class heuristic (3+ children, no attributes defined)
        _children_count = {}
        for (_child, _parent) in _gen_edges:
            _children_count[_parent] = _children_count.get(_parent, 0) + 1

        for _cls_lbl in class_labels:
            _cls_n = _normalize(_cls_lbl)
            _cnt   = _children_count.get(_cls_n, 0)
            if _cnt >= 3:
                _cls_shape = next(
                    (s for s in class_shapes
                     if _normalize(_get_class_label(s)) == _cls_n), None
                )
                _has_attrs = False
                if _cls_shape:
                    _attrs_raw = str(_cls_shape.get(
                        "attributes", _cls_shape.get(
                        "attrs",      _cls_shape.get(
                        "properties", ""))
                    )).strip()
                    _has_attrs = bool(
                        _attrs_raw and
                        _attrs_raw.lower() not in ("none","null","undefined","[]","{}","")
                    )
                if not _has_attrs:
                    errors.append(ValidationError(
                        error_type  = "SEMANTIC_ABSTRACT_SUGGESTION",
                        description = (
                            f"Class '{_cls_lbl}' has {_cnt} child classes "
                            f"(via generalization) but no attributes are defined. "
                            f"Consider marking it as abstract in UML using the "
                            f"{{abstract}} stereotype or italicized class name."
                        ),
                        suggestion  = (
                            f"Add the {{abstract}} stereotype to '{_cls_lbl}', "
                            f"or define shared attributes that its child classes inherit."
                        ),
                        severity    = ValidationError.SEVERITY_INFO,
                        element     = _cls_lbl,
                    ))

        #  Score & summary
        # ═════════════════════════════════════════════════════════════════════
        error_count   = len([e for e in errors if e.severity == "ERROR"])
        warning_count = len([e for e in errors if e.severity == "WARNING"])
        info_count    = len([e for e in errors if e.severity == "INFO"])
        score = max(0, 100 - (error_count * 15) - (warning_count * 5))
        # Success = zero ERRORs. Warnings/INFO are advisory only.
        if error_count == 0:
            summary = "Diagram is Correct 🎉"
            status  = "correct"
        else:
            summary = f"❌ {error_count} error(s), {warning_count} warning(s) found."
            status  = "errors"
        result = self._build_result(errors, score, summary)
        # Attach extra fields so frontend can show green panel
        result["status"]  = status
        result["summary"] = summary      # ensure "summary" key always present
        result["message"] = summary      # alias in case frontend uses "message"
        result["isCorrect"] = (status == "correct")
        return result
    # ── Helpers ───────────────────────────────────────────────────────────────
    def _get_relationships(self, shapes: List[Dict]) -> List[Dict[str, str]]:
        """
        Extract relationships from shapes.
        Handles ALL known key-name variants that different diagram tools produce.

        PRIMARY resolution: named keys (from/to, source/target, etc.)
        FALLBACK resolution: spatial proximity — find the nearest classShape
          to the line's start-point and end-point using canvas coordinates.

        A shape is treated as a relationship if its type contains a relationship
        keyword, OR it has both a source-end AND a target-end field populated.
        """
        import math as _math

        relationships = []
        rel_keywords = (
            "arrow", "line", "connect", "association", "relation",
            "generalization", "inheritance", "aggregation",
            "composition", "dependency", "link", "edge",
        )
        CLASS_TYPE_KEYWORDS = (
            "class", "rectangle", "rect", "box", "entity",
            "node", "umlclass", "classshape", "class_shape", "shape",
        )

        # Markers that Flutter injects between class name / attrs / ops sections
        _SECTION_MARKERS = ("---attrs---", "---ops---", "---methods---",
                            "---attributes---", "---operations---")

        def _clean_label(raw: str) -> str:
            """
            Extract the real class name from a potentially multiline label.
            Flutter sometimes sends the full classShape text block as the
            from/to value, e.g.:
              "Bank\n---attrs---\nbank name : String\n---ops---"
            We want just "Bank".
            """
            if not raw:
                return ""
            # If any section marker is present, take only the text before it
            lower = raw.lower()
            for marker in _SECTION_MARKERS:
                idx = lower.find(marker)
                if idx != -1:
                    raw = raw[:idx]
            # Also strip at the first newline
            first_line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
            return first_line

        def _first(*args):
            """Return first non-empty string from candidates (cleaned)."""
            for a in args:
                v = _clean_label(str(a)) if a is not None else ""
                if v and v.lower() not in ("none", "null", "undefined", ""):
                    return v
            return ""

        # ── Build a list of class-shapes with their canvas centre positions ──
        def _get_pos(shape_dict, key):
            pos = shape_dict.get(key)
            if isinstance(pos, dict):
                try:
                    x = float(pos.get("dx", pos.get("x", 0)) or 0)
                    y = float(pos.get("dy", pos.get("y", 0)) or 0)
                    return x, y
                except (TypeError, ValueError):
                    return None
            return None

        def _looks_like_class(s):
            """True if shape should be treated as a class node (not a line/arrow)."""
            stype = str(s.get("type", s.get("shapeType", s.get("shape_type", "")))).lower()
            if any(k in stype for k in rel_keywords):
                return False
            if any(k in stype for k in CLASS_TYPE_KEYWORDS):
                return True
            # Semantic fallback: labelled shape without an endPosition → treat as class
            has_label = any(
                str(s.get(k, "")).strip() not in ("", "none", "null", "undefined")
                for k in ("text", "name", "label", "title", "className",
                          "class_name", "caption", "value", "content")
            )
            return has_label and s.get("endPosition") is None

        class_shapes_pos = []   # list of (label, cx, cy)
        for s in shapes:
            # Must look like a class and NOT a relationship
            if not _looks_like_class(s):
                continue
            label = ""
            for key in ("className", "class_name", "header", "title", "name",
                        "label", "caption", "text", "value", "content"):
                v = _clean_label(str(s.get(key, "")).strip())
                if v and v.lower() not in ("none", "null", "undefined", ""):
                    label = v
                    break
            if not label:
                continue
            pos = _get_pos(s, "position")
            if pos is None:
                continue
            px, py = pos
            # Centre = position + half size
            size = s.get("size") or {}
            if isinstance(size, dict):
                w = size.get("width", 150)
                h = size.get("height", 100)
            else:
                w, h = 150, 100
            cx = px + w / 2
            cy = py + h / 2
            class_shapes_pos.append((label, cx, cy))

        def _nearest_class(px, py):
            """Return the label of the classShape whose centre is closest to (px,py)."""
            if not class_shapes_pos:
                return ""
            best_label = ""
            best_dist  = float("inf")
            for (lbl, cx, cy) in class_shapes_pos:
                dist = _math.hypot(cx - px, cy - py)
                if dist < best_dist:
                    best_dist  = dist
                    best_label = lbl
            return best_label

        for s in shapes:
            shape_type = _first(
                s.get("type"), s.get("shapeType"), s.get("shape_type"), s.get("kind"), ""
            ).lower()
            is_rel_type = any(k in shape_type for k in rel_keywords)

            # ── Primary: named from/to keys ───────────────────────────────────
            frm = _first(
                s.get("from"),       s.get("source"),      s.get("startId"),
                s.get("startText"),  s.get("fromClass"),   s.get("sourceClass"),
                s.get("startLabel"), s.get("fromLabel"),   s.get("start"),
                s.get("sourceNode"), s.get("fromNode"),
            )
            to = _first(
                s.get("to"),         s.get("target"),      s.get("endId"),
                s.get("endText"),    s.get("toClass"),     s.get("targetClass"),
                s.get("endLabel"),   s.get("toLabel"),     s.get("end"),
                s.get("targetNode"), s.get("toNode"),
            )

            # ── Fallback: position-based class resolution ─────────────────────
            # Used when the Flutter app sends from/to as empty strings but
            # includes canvas coordinates (position + endPosition).
            if is_rel_type and (not frm or not to):
                start_pos = _get_pos(s, "position")
                end_pos_rel = _get_pos(s, "endPosition")   # relative offset
                if start_pos and end_pos_rel:
                    sx, sy = start_pos
                    ex = sx + end_pos_rel[0]
                    ey = sy + end_pos_rel[1]
                    if not frm:
                        frm = _nearest_class(sx, sy)
                    if not to:
                        to  = _nearest_class(ex, ey)

            has_endpoints = bool(frm and to)
            if not (is_rel_type or has_endpoints):
                continue

            # ── Multiplicities ────────────────────────────────────────────────
            mult_from = _first(
                s.get("multiplicityFrom"),    s.get("multiplicity_from"),
                s.get("sourceMult"),          s.get("multFrom"),
                s.get("startMultiplicity"),   s.get("fromMultiplicity"),
                s.get("labelFrom"),           s.get("sourceMultiplicity"),
                s.get("multiplicityStart"),   s.get("mult_from"),
                s.get("mult1"),
            )
            mult_to = _first(
                s.get("multiplicityTo"),      s.get("multiplicity_to"),
                s.get("targetMult"),          s.get("multTo"),
                s.get("endMultiplicity"),     s.get("toMultiplicity"),
                s.get("labelTo"),             s.get("targetMultiplicity"),
                s.get("multiplicityEnd"),     s.get("mult_to"),
                s.get("mult2"),
            )

            # Parse multiplicity from text field if still missing
            # Flutter encodes it as "startMult|label|endMult"
            if not mult_from and not mult_to:
                raw_text = _first(s.get("text", ""))
                if raw_text:
                    parts = raw_text.split("|")
                    if len(parts) >= 3:
                        mult_from = parts[0].strip()
                        mult_to   = parts[2].strip()
                    elif len(parts) == 2:
                        mult_from = parts[0].strip()
                        mult_to   = parts[1].strip()

            # Generic combined multiplicity field fallback
            if not mult_from and not mult_to:
                combined = _first(
                    s.get("multiplicity"), s.get("mult")
                )
                if combined:
                    parts = re.split(r"[,/|]", combined)
                    if len(parts) == 2:
                        mult_from = parts[0].strip()
                        mult_to   = parts[1].strip()
                    else:
                        mult_from = combined
                        mult_to   = ""

            # ── Association / relationship label (middle label on the line) ────
            # Flutter may encode this in several ways:
            #   (a) a dedicated field like "label", "name", "relationName", etc.
            #   (b) in the "text" field as "startMult|RelationLabel|endMult"
            #       (already parsed above — take parts[1] if len==3)
            rel_label = _first(
                s.get("label"),           s.get("name"),
                s.get("relationName"),    s.get("relation_name"),
                s.get("assocLabel"),      s.get("assoc_label"),
                s.get("associationName"), s.get("association_name"),
                s.get("roleName"),        s.get("role_name"),
                s.get("linkLabel"),       s.get("link_label"),
                s.get("edgeLabel"),       s.get("edge_label"),
            )
            # Reject label if it looks like a multiplicity value (e.g. "1", "0..*")
            if rel_label and VALID_MULTIPLICITY_RE.match(rel_label.strip()):
                rel_label = ""
            # If still empty, try "text" field parsed as pipe-separated triple
            if not rel_label:
                raw_text_label = _first(s.get("text", ""))
                if raw_text_label:
                    parts_lbl = raw_text_label.split("|")
                    if len(parts_lbl) == 3:
                        rel_label = parts_lbl[1].strip()  # middle part is the label

            if frm and to:
                relationships.append({
                    "from":              frm,
                    "to":                to,
                    "type":              shape_type,
                    "multiplicity_from": mult_from,
                    "multiplicity_to":   mult_to,
                    "label":             rel_label,
                })
        return relationships
