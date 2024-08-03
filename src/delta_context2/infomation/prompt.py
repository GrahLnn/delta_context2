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

PARAGRAPH_ALIGNMENT_TO_SENTENCE_PROMPT = """
    You are tasked with separating two paragraphs in different languages that contain the same content. Your goal is to match smaller sentences from each paragraph and present them in a specific JSON format. If a corresponding sentence cannot be found, you should add an appropriate translation based on the content and context.

Here are the two paragraphs:

Paragraph A:
<paragraph_a>
{PARAGRAPH_A}
</paragraph_a>

Paragraph B:
<paragraph_b>
{PARAGRAPH_B}
</paragraph_b>

Follow these steps to complete the task:

1. Read both paragraphs carefully.
2. Identify corresponding sentences or phrases in both paragraphs that convey the same meaning.
3. Split the paragraphs into smaller, matching sentences or phrases.
4. If a corresponding sentence cannot be found in one paragraph, create an appropriate translation based on the content and context.
5. Create a JSON object with a "pair" array containing objects for each matched sentence pair.
6. Each object in the "pair" array should have two properties: "sentence_a" for the sentence from Paragraph A, and "sentence_b" for the corresponding sentence from Paragraph B (or the translation you created).

Format your response as a JSON object with the following structure:

```json
{{
  "pair": [
    {{
      "sentence_a": "Sentence from Paragraph A",
      "sentence_b": "Corresponding sentence from Paragraph B or translation"
    }},
    {{
      "sentence_a": "Next sentence from Paragraph A",
      "sentence_b": "Next corresponding sentence from Paragraph B or translation"
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
      "sentence_b": "" // Assuming this sentence is not found in Paragraph B, it should be empty, so the sentence_b here is empty string "".
    }},
    {{
      "sentence_a": "The weather is nice today.",
      "sentence_b": "El clima está agradable hoy." 
    }}
  ]
}}
```

In this example, the last pair demonstrates a case where a direct corresponding sentence might not have been present in Paragraph B, so an appropriate translation was provided.

Please process the given paragraphs and provide your answer in the specified JSON format. Ensure that the sentences are properly matched, translations are added where necessary, and that the JSON is correctly formatted. Your entire response should be enclosed in ``` tags."""

SINGLE_TRANSLATION_PROMPT = """You are tasked with rewriting a given text in engaging Chinese sentences. Your goal is to create a version that is captivating and interesting to Chinese readers while maintaining the original meaning. Follow these steps:

1. Here is the original text you will be working with:

<original_text>
{ORIGINAL_TEXT}
</original_text>

2. Rewrite the text in Chinese, focusing on creating engaging and interesting sentences. Use vivid language, appropriate idioms, and cultural references when applicable to make the text more appealing to Chinese readers.

3. Ensure that you maintain the core meaning and key points of the original text while adapting it for a Chinese audience.

4. Use appropriate Chinese characters and grammar. Make sure your writing is fluent and natural-sounding to native Chinese speakers.

5. Only provide the rewritten Chinese text as your output. Do not include any explanations, comparisons to the original text, or additional comments.

6. Present your rewritten text within <chinese_text> tags.

Remember, your task is to create an engaging Chinese version of the text, not a literal translation. Focus on conveying the essence of the original content in a way that resonates with Chinese readers."""

SUMMARY_SYS_MESSAGE = 'You are a summarizing assistant responsible for analyzing the content of YouTube videos. The user will feed you transcriptions but you should always refer to the content in your response as "the video". Focus on accurately summarizing the main points and key details of the videos. Do not comment on the style of the video (e.g., whether it is a voiceover or conversational). Do never mention or imply the existence of text, transcription, or any written format. Use phrases like "The video discusses..." or "According to the video...". Strive to be the best summarizer possible, providing clear, and informative summaries that exclusively reference the video content.'

SPLIT_SMALL_SENTENCE_PROMPT = "Divide the following sentence into {PARTS_NUM} parts, connected by line break and return nothing else.\n\n{TEXT}"

TRANSCRIBTION_CORECTION_PROMPT = """You are tasked with improving a transcribed text by unifying terminology, correcting spelling errors, and ensuring content consistency and accuracy of details. This task is crucial for maintaining clarity and professionalism in written communications. Here is the transcribed text you will be working with:

<transcribed_text>
{TRANSCRIBED_TEXT}
</transcribed_text>

Follow these steps to complete the task:

1. Spelling Correction:
   - Carefully read through the entire text.
   - Identify and correct any spelling errors you encounter.
   - Pay special attention to proper nouns, technical terms, and industry-specific jargon.

2. Terminology Unification:
   - Look for instances where different terms are used to refer to the same concept, object, or person.
   - Choose the most appropriate term and use it consistently throughout the text.
   - If multiple terms are equally valid, select the one used most frequently in the original text.

3. Content Consistency:
   - Ensure that facts, figures, and details are consistent throughout the document.
   - If you encounter conflicting information, use your best judgment to determine the correct version based on context.

4. Accuracy Preservation:
   - While making corrections, be careful not to alter the original meaning or intent of the text.
   - If you're unsure about a particular correction, err on the side of caution and maintain the original wording.

5. Final Review:
   - After making all necessary corrections and unifications, read through the entire text once more to ensure coherence and flow.
   - Make any final adjustments needed for clarity and consistency.
   - Transcripts come from videos, so they retain some informal language and additional notes to preserve the original style.
   - This is a preprocessing task for subtitles, so there's no need to determine if there are any redundant parts.
   - If there are any unclear parts, insert the [inaudible] marker.

6. Output:
   - Provide the corrected and unified text within <repaired_transcript> tags.
   - If you made any significant changes or encountered any notable issues, briefly explain them within <notes> tags after the repaired transcript.
   - No need to omit any paragraphs even it's seems to be irrelevant.

Remember, your goal is to improve the text while preserving its original meaning and intent. Make only necessary changes to enhance clarity, consistency, and professionalism."""
