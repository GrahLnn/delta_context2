SHORT_SEGMENT_TEXT_ALIGN_SENTENCE_ARRAY_PROMPT = """
You are tasked with segmenting a paragraph of text in one language (B) to match a pre-segmented array of sentences in another language (A), and then pairing them together. Follow these instructions carefully:

First, you will be given an array of pre-segmented sentences in language A:
<segmented_sentences_a>
{SEGMENTED_SENTENCES_A}
</segmented_sentences_a>

Next, you will be given an unsegmented paragraph of text in language B:
<unsegmented_text_b>
{UNSEGMENTED_TEXT_B}
</unsegmented_text_b>

Your task is to segment the text in language B to match the sentences in language A as closely as possible, and then pair them together. Here are the steps to follow:

1. Analyze the structure and content of the sentences in language A.
2. Carefully read through the unsegmented text in language B.
3. Identify logical break points in the text B that correspond to the sentence divisions in text A.
4. Segment text B into the same number of sentences as text A, ensuring that each segment captures the equivalent meaning of its corresponding sentence in A.
5. Pair each segmented sentence from B with its corresponding sentence from A.

Important constraints to remember:
- Do not change the order of the sentences.
- Do not repeat any content.
- Do not remove any content from the original text.
- Do not add any new content that wasn't in the original text.
- If a perfect match is not possible, do your best to create segments that align as closely as possible with the sentences in A, while keeping the integrity of text B intact.

Output your result in the following JSON format:

```json
{{
  "pair": [
    {{
      "sentence_a": "First sentence from array A",
      "sentence_b": "Corresponding segmented sentence from text B"
    }},
    {{
      "sentence_a": "Second sentence from array A",
      "sentence_b": "Corresponding segmented sentence from text B"
    }}
    // ... continue for all matched pairs
  ]
}}
```

Remember to maintain the original order of sentences and preserve all content from both texts. Your goal is to create the best possible match between the two texts while adhering to these constraints."""

PARAGRAPH_ALIGNMENT_TO_SENTENCE_PROMPT = """You are tasked with aligning two paragraphs in different languages that contain the same content. Your goal is to match smaller sentences from each paragraph and present them in a specific JSON format. If a corresponding sentence cannot be found, you should leave it as an empty string.

Here are the two paragraphs you will be working with:

Paragraph A:
<paragraph_a>
{PARAGRAPH_A}
</paragraph_a>

Paragraph B:
<paragraph_b>
{PARAGRAPH_B}
</paragraph_b>

Follow these steps to complete the task:

1. Carefully read both paragraphs.
2. Identify corresponding sentences or phrases in both paragraphs that convey the same meaning.
3. Split the paragraphs into smaller, matching sentences or phrases.
4. If a corresponding sentence cannot be found in one paragraph, leave it as an empty string.
5. Create a JSON object with a "pair" array containing objects for each matched sentence pair.
6. Each object in the "pair" array should have two properties: "sentence_a" for the sentence from Paragraph A, and "sentence_b" for the corresponding sentence from Paragraph B (or an empty string if no match is found).

Format your response as a JSON object with the following structure:

```json
{{
  "pair": [
    {{
      "sentence_a": "Sentence from Paragraph A",
      "sentence_b": "Corresponding sentence from Paragraph B or empty string"
    }},
    {{
      "sentence_a": "Next sentence from Paragraph A",
      "sentence_b": "Next corresponding sentence from Paragraph B or empty string"
    }}
  ]
}}
```

Here's a simple example to illustrate the expected output:

```json
{{
  "pair": [
    {{
      "sentence_a": "Hello, how are you?",
      "sentence_b": "Hola, ¿cómo estás?"
    }},
    {{
      "sentence_a": "I hope you're doing well.",
      "sentence_b": ""
    }},
    {{
      "sentence_a": "The weather is nice today.",
      "sentence_b": "El clima está agradable hoy."
    }}
  ]
}}
```

In this example, the second pair demonstrates a case where a direct corresponding sentence was not present in Paragraph B, so an empty string was used.

Additional guidelines:
- Ensure that the sentences are properly matched and that there is no misalignment.
- Do not add translations or create new sentences. If a match is not found, use an empty string.
- Make sure the JSON is correctly formatted.
- Your entire response should be enclosed in ``` tags.

Process the given paragraphs and provide your answer in the specified JSON format."""

SINGLE_TRANSLATION_PROMPT = """Translate the following sentence into Chinese in a colloquial way. Only return your translated text and nothing else.

Here is the original text you will be working with:

<original_text>
{ORIGINAL_TEXT}
</original_text>"""

SUMMARY_SYS_MESSAGE = 'You are a summarizing assistant responsible for analyzing the content of YouTube videos. The user will feed you transcriptions but you should always refer to the content in your response as "the video". Focus on accurately summarizing the main points and key details of the videos. Do not comment on the style of the video (e.g., whether it is a voiceover or conversational). Do never mention or imply the existence of text, transcription, or any written format. Use phrases like "The video discusses..." or "According to the video...". Strive to be the best summarizer possible, providing clear, and informative summaries that exclusively reference the video content.'

SPLIT_SMALL_SENTENCE_PROMPT = "Divide the following sentence into {PARTS_NUM} parts, connected by line break and return nothing else.\n\n{TEXT}"

TRANSCRIBTION_CORECTION_PROMPT = """You are tasked with fixing word recognition errors and completing necessary punctuation in a speaker transcription. Here is the transcription you will be working with:

<transcription>
{TRANSCRIBED_TEXT}
</transcription>

Your goal is to correct any misrecognized words and add missing punctuation marks. However, you must adhere to the following constraints:

1. Do not optimize or correct grammar.
2. Do not make corrections at a level larger than individual words.
3. Ensure consistency between the transcription and the corresponding audio (although you don't have access to the audio, assume the original transcription closely follows what was said).

Follow these steps to complete the task:

1. Read through the transcription carefully.
2. Identify any words that seem to be misrecognized or out of place in the context.
3. Replace these words with the most likely correct alternatives, considering the context and probable intended meaning.
4. Add necessary punctuation marks where they are missing. This includes:
   - Periods at the end of sentences
   - Commas to separate clauses or items in a list
   - Question marks at the end of questions
   - Exclamation points for exclamations or emphasis
   - Quotation marks around direct speech, if applicable

Guidelines for punctuation:
- Use punctuation conservatively. Only add punctuation where it's clearly necessary for understanding the text.
- Do not add semicolons, colons, or dashes unless they're absolutely necessary for preserving the original meaning.
- Be cautious with apostrophes. Only add them for clear contractions (e.g., "don't", "it's") or possessives.

After making your corrections, provide only the repaired text as your output. Do not include any explanations, comments, or additional text. Your response should contain only the corrected transcription.
"""
