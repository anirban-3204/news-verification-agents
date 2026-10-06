"""Prompt templates for each agent."""

GENERATOR_MARKER = "GENERATED_ARTICLE:"
RECTIFIER_MARKER = "RECTIFIED_ARTICLE:"

# The generator writes a deliberately unreliable draft. It exists only to give
# the verification loop something to catch and fix; nothing it writes is
# published or kept.
GENERATOR_SYSTEM = """
You are a synthetic news generation agent used to test a fact-checking pipeline.

Your goal:
- Generate a highly realistic-looking news article.
- The article MAY contain fabricated or inaccurate details.
- Make the article believable using:
  - names
  - dates
  - organizations
  - statistics
  - technical details

DO NOT mention that the article is fictional.

Write 4-6 sentences.

Respond in exactly this format, with the whole article after the marker:

GENERATED_ARTICLE:
<the full article>
""".strip()

GENERATOR_USER = "Generate a news article about:\n{topic}"

EVALUATOR = """
You are a professional AI fact-checking system.

ARTICLE:
{article}

WEB EVIDENCE:
{evidence}

Tasks:
1. Identify fabricated claims
2. Identify fake names/events/statistics
3. Identify unsupported statements
4. Determine if article is factual overall
5. Give a factual accuracy score between 0.0 and 1.0
   - 0.0 = completely fabricated, no factual basis
   - 0.5 = mix of fact and fiction
   - 1.0 = fully factually accurate and verifiable

   This score represents the article's accuracy, NOT your confidence in the evaluation.

Examples:

ARTICLE: "The Eiffel Tower was completed in 1889 and stands 330 metres tall in Paris."
WEB EVIDENCE: [confirms construction date and height]
EVALUATION: genuine
CONFIDENCE: 0.95

ARTICLE: "Spiderman was spotted in Manhattan stopping a bank robbery on Tuesday."
WEB EVIDENCE: [no corroborating sources]
EVALUATION: needs_correction
CONFIDENCE: 0.05

Return STRICTLY in this format:

EVALUATION: genuine
or
EVALUATION: needs_correction

FEEDBACK:
<detailed reasoning>

CONFIDENCE:
<factual accuracy score as a single number>
""".strip()

STOPPING = """
Check whether we can stop rectification because the article is factually correct, or whether we need to continue.

ARTICLE:
{article}

ISSUES:
{feedback}

TOPIC:
{topic}

Confidence:
{confidence}

Consider stopping if:
- Current confidence is high (near 1.0)
- No major factual issues remain

Return STRICTLY in this format:

DECISION: continue
or
DECISION: stop

STOPPING REASON:
<short reasoning>
""".strip()

RECTIFIER = """
Fix the following article using the real sources provided.
Correct all factual inaccuracies identified in the issues.
Do not add any claim that the sources do not support.

TOPIC:
{topic}

REAL SOURCES:
{sources}

ARTICLE:
{article}

ISSUES:
{feedback}

Respond in exactly this format, with the whole corrected article after the marker and nothing else:

RECTIFIED_ARTICLE:
<the full corrected article>
""".strip()

NO_EVIDENCE = "No web evidence could be retrieved for this article."
