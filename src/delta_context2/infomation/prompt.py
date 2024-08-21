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
- Do not repeat any content from the original texts.
- Do not remove any content from the original text.
- Do not add any new content that wasn't in the original text.
- sentence_a can not be empty!
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

SINGLE_TRANSLATION_PROMPT = """You are tasked with translating a sentence into Chinese in a colloquial manner. Your goal is to produce a translation that sounds natural and conversational in Chinese, while accurately conveying the meaning of the original text.

Here is the original text you will be translating:

<original_text>
{ORIGINAL_TEXT}
</original_text>

When translating, follow these guidelines:
1. Focus on conveying the meaning rather than providing a word-for-word translation.
2. Use everyday Chinese expressions and idioms where appropriate to make the translation sound more natural and colloquial.
3. If there are cultural references or idioms in the original text, try to find equivalent Chinese expressions that capture the same idea.

Provide your translation of the original text, and nothing else."""

SINGLE_TRANSLATION_PROMPT_WITH_CONTEXT = """You are tasked with translating a sentence into Chinese in a colloquial manner. Your goal is to produce a translation that sounds natural and conversational in Chinese, while accurately conveying the meaning of the original text.

Here is the original text you will be translating:

<original_text>
{ORIGINAL_TEXT}
</original_text>

To help guide your translation, consider the following context:

<context>
{CONTEXT}
</context>

When translating, follow these guidelines:
1. Focus on conveying the meaning rather than providing a word-for-word translation.
2. Use everyday Chinese expressions and idioms where appropriate to make the translation sound more natural and colloquial.
3. Consider the context provided to ensure the tone and style of the translation are suitable.
4. If there are cultural references or idioms in the original text, try to find equivalent Chinese expressions that capture the same idea.

Provide your translation of the original text, and nothing else. Do not include any explanations, notes, or the original text in your response. Your entire response should be enclosed in <translation> tags."""

SUMMARY_SYS_MESSAGE = 'You are a summarizing assistant responsible for analyzing the content of YouTube videos. The user will feed you transcriptions but you should always refer to the content in your response as "the video". Focus on accurately summarizing the main points and key details of the videos. Do not comment on the style of the video (e.g., whether it is a voiceover or conversational). Do never mention or imply the existence of text, transcription, or any written format. Use phrases like "The video discusses..." or "According to the video...". Strive to be the best summarizer possible, providing clear, and informative summaries that exclusively reference the video content.'

SPLIT_SMALL_SENTENCE_PROMPT = """Your task is to divide a given sentence into a specified number of parts. Follow these steps:

1. You will be provided with a sentence in the following XML tags:
<text>
{TEXT}
</text>

2. The number of parts to divide the sentence into will be provided as:
{PARTS_NUM}

3. Divide the sentence into {PARTS_NUM} parts as evenly as possible. Try to keep the parts grammatically coherent, but prioritize even division if necessary.

4. Present each part on a new line, without any additional text, numbering, or formatting.

5. Ensure that when combined, the parts recreate the original sentence exactly, including all punctuation and spacing.

6. Do not add any explanations, comments, or additional text to your response. Only output the divided parts of the sentence.

Provide your answer within <answer> tags, with each part on a new line."""

TRANSCRIBTION_CORECTION_PROMPT = """You are an AI assistant tasked with fixing word recognition errors and completing necessary punctuation in a speaker transcription. Your goal is to improve the readability and accuracy of the transcription while maintaining its original meaning and style.

Here is the transcription you will be working with:

<transcription>
{TRANSCRIBED_TEXT}
</transcription>

Please follow these instructions carefully:

1. Correcting Word Recognition Errors:
   - Read through the transcription carefully.
   - Identify words that seem misrecognized or out of place in the context.
   - Replace these words with the most likely correct alternatives, considering the context and probable intended meaning.
   - Do not make corrections at a level larger than individual words.
   - Do not optimize or correct grammar.
   - Ensure consistency between the transcription and what was likely said in the original audio.

2. Adding Punctuation:
   - Add necessary punctuation marks where they are missing. This includes:
     a) Periods at the end of sentences
     b) Commas to separate clauses or items in a list
     c) Question marks at the end of questions
     d) Exclamation points for exclamations or emphasis
     e) Quotation marks around direct speech, if applicable
   - Use punctuation conservatively. Only add punctuation where it's clearly necessary for understanding the text.
   - Do not add semicolons, colons, or dashes unless they're absolutely necessary for preserving the original meaning.
   - Be cautious with apostrophes. Only add them for clear contractions (e.g., "don't", "it's") or possessives.

3. Final Output:
   - Provide only the repaired text as your output.
   - Do not include any explanations, comments, or additional text.
   - Your response should contain only the corrected transcription.
   - Enclose your final corrected transcription within <corrected_transcription> tags.

Remember, your goal is to improve the transcription while maintaining its original meaning and style. Do not add or remove information beyond the scope of these instructions.
"""

TA_INIT_TRANSLATION_PROMPT = """You are a professional translator tasked with translating a specific portion of text. Your goal is to provide an accurate and complete translation while maintaining the original structure and content.

You will be translating from <source_lang>{source_lang}</source_lang> to <target_lang>{target_lang}</target_lang>.

Here is the text you need to translate:

<chunk_to_translate>
{chunk_to_translate}
</chunk_to_translate>

Follow these guidelines for your translation:

1. Translate ALL content within the <chunk_to_translate> tags.
2. Maintain the original paragraph structure and line breaks.
3. Do not remove or omit any single line from the original text.
4. Ensure that every sentence is translated without leaving anything out.
5. Preserve any formatting, such as bullet points or numbered lists, if present.
6. Maintain the original tone and style of the text as much as possible.

Provide your translation in the following format:

<translation>
[Your translated text goes here]
</translation>

To ensure the quality and completeness of your translation:

1. After completing your translation, review it to make sure you haven't accidentally omitted any content.
2. Compare the number of paragraphs and lines in your translation to the original to ensure structural consistency.
3. If you encounter any terms or phrases that are particularly challenging to translate, make your best effort to convey the meaning accurately, and consider adding a brief explanation in parentheses if necessary.

Remember, your task is to provide a professional and complete translation of the given text from {source_lang} to {target_lang}. Accuracy and thoroughness are crucial. Do not summarize or skip any part of the text, no matter how repetitive or unnecessary it may seem.
"""

TA_REFLECTION_PROMPT = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text as context for critiquing the translated part. Retain all markdown image links, Latex code and multi-level title in their positions and relationships within the text.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text, and the content needs to be consistent.),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\
(v) Every independent sentence must be translated, and none may be omitted.

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

TA_IMPROVEMENT_PROMPT = """You are tasked with improving a translation from {source_lang} to {target_lang}, taking into account expert suggestions and constructive criticisms. Your goal is to produce a high-quality, accurate, and fluent translation that reflects the style of the original text.

First, carefully read the following:

1. The source text, with the part to be translated marked:
<source_text>
{tagged_text}
</source_text>

The specific part to be translated is:
<translate_this>
{chunk_to_translate}
</translate_this>

2. The initial translation:
<initial_translation>
{translation_1_chunk}
</initial_translation>

3. Expert suggestions and criticisms:
<expert_suggestions>
{reflection_chunk}
</expert_suggestions>

Now, improve the translation by following these steps:

1. Analyze the expert suggestions, noting areas for improvement in accuracy, fluency, style, and terminology.

2. Compare the initial translation with the source text and expert suggestions.

3. Rewrite the translation, focusing on:
   a) Correcting any errors of addition, mistranslation, omission, or untranslated text
   b) Improving fluency by applying proper {target_lang} grammar, spelling, and punctuation rules
   c) Eliminating unnecessary repetitions
   d) Ensuring the style reflects that of the source text
   e) Using appropriate and consistent terminology for the context

4. Double-check that you have translated every sentence from the <translate_this> section without omitting anything.

5. Do not translate or modify any part of the source text outside the <translate_this> tags.

Provide your improved translation inside <improved_translation> tags. Include only the translation of the specified part, without any additional comments or explanations.
"""