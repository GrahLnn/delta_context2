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

PARAGRAPH_ALIGNMENT_TO_SENTENCE_PROMPT = """You are an AI assistant specializing in aligning multilingual text. Your task is to match corresponding sentences or phrases between two paragraphs in different languages and present them in a specific JSON format.

Here are the two paragraphs you will be working with:

<paragraph_a>
{PARAGRAPH_A}
</paragraph_a>

<paragraph_b>
{PARAGRAPH_B}
</paragraph_b>

Follow these steps to complete the task:

1. Carefully analyze both paragraphs.
2. Identify individual sentences or phrases in each paragraph.
3. Match corresponding sentences or phrases between Paragraph A and Paragraph B.
4. Create a JSON object with the matched pairs.

Your output should be a JSON object with the following structure:

```json
{{
  "pair": [
    {{
      "sentence_a": "Sentence from Paragraph A",
      "sentence_b": "Corresponding sentence from Paragraph B or empty string"
    }},
    {{
      "sentence_a": "Another sentence from Paragraph A",
      "sentence_b": "Another corresponding sentence from Paragraph B or empty string"
    }}
  ]
}}
```

Important rules to follow:
- Maintain the original order of sentences strictly.
- Ensure every sentence from Paragraph A has a corresponding entry in the JSON structure.
- Use all sentences from Paragraph B.
- Do not cross-match or reorder sentences.
- If you are unsure about a match, use an empty string for sentence_b instead of an incorrect match.
- If sentence_a cannot be matched with a correct translation, you must leave sentence_b empty. Do not include any incorrect content in sentence_b.

Provide your final output as a valid JSON object inside ```json tags. Do not include any explanations or additional text outside of these tags."""

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

Provide your translation of the original text, and nothing else. Do not include any explanations, notes, or the original text in your response."""

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

You will be translating from <source_lang>{SOURCE_LANG}</source_lang> to <target_lang>{TARGET_LANG}</target_lang>.

Here is the text you need to translate:

<chunk_to_translate>
{CHUNK_TO_TRANSLATE}
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

Remember, your task is to provide a professional and complete translation of the given text from {SOURCE_LANG} to {TARGET_LANG}. Accuracy and thoroughness are crucial. Do not summarize or skip any part of the text, no matter how repetitive or unnecessary it may seem.
"""

TA_REFLECTION_PROMPT = """You are a professional translator and language expert. Your task is to carefully read a source text and part of its translation, then provide constructive criticism and helpful suggestions for improving the translation. The goal is to ensure the translation is accurate, fluent, and stylistically appropriate for the target language and region.

Here's the source text, with the part to be translated marked:

<source_text>
{tagged_text}
</source_text>

The specific part to be translated is:

<translate_this>
{chunk_to_translate}
</translate_this>

Now, here's the translation of that part that you need to critique:

<translation>
{translation_1_chunk}
</translation>

When evaluating the translation, consider the following aspects:

1. Accuracy: Check for errors of addition, mistranslation, omission, or untranslated text. Ensure the content is consistent with the source.
2. Fluency: Verify that the translation follows {target_lang} grammar, spelling, and punctuation rules. Look for unnecessary repetitions.
3. Style: Ensure the translation reflects the style of the source text and takes into account the cultural context of {country}.
4. Terminology: Check that terminology use is consistent and reflects the source text domain. Use equivalent idioms in {target_lang} where appropriate.
5. Completeness: Verify that every independent sentence has been translated, with none omitted.

Provide a list of specific, helpful, and constructive suggestions for improving the translation. Each suggestion should address one specific part of the translation. Format your suggestions as follows:

<suggestions>
1. [Brief description of the issue]: [Explanation and suggestion for improvement]
2. [Brief description of the issue]: [Explanation and suggestion for improvement]
[Continue with numbered suggestions as needed]
</suggestions>

Important: As this is a podcast transcript, it's crucial to check if any sentences are missing. Every statement made by the speaker needs to be retained. If you notice any omissions, provide suggestions for translating the missing content using the same format as above.

Your final output should contain only the numbered list of suggestions within the <suggestions> tags. Do not include any other text or explanations outside of these tags."""

TA_IMPROVEMENT_PROMPT = """You are tasked with improving a translation from {source_lang} to {target_lang}, focusing on readability and fluency while maintaining accuracy. Your goal is to produce a high-quality translation that is easy to read and understand.
First, carefully read the following:

The source text, with the part to be translated marked:
<source_text>
{tagged_text}
</source_text>

The specific part to be translated is:
<translate_this>
{chunk_to_translate}
</translate_this>

The initial translation:
<initial_translation>
{translation_1_chunk}
</initial_translation>
Expert suggestions and criticisms:
<expert_suggestions>
{reflection_chunk}
</expert_suggestions>

Now, improve the translation by following these steps:

Analyze the expert suggestions, focusing on areas that can improve readability and fluency.
Compare the initial translation with the source text and expert suggestions.
Rewrite the translation, focusing on:
a) Improving sentence structure and flow
b) Using natural and idiomatic {target_lang} expressions
c) Ensuring clarity and coherence
d) Maintaining the original meaning and tone
e) You are not allowed to add any additional text, extra comments or explanations to your response.
Double-check that you have translated every sentence from the <translate_this> section without omitting or adding information.
Do not translate or modify any part of the source text outside the <translate_this> tags.
f) Remove any pinyin annotations.


<improved_translation>
[Your improved translation goes here]
</improved_translation>

Ensure that your improved translation addresses all the expert suggestions and adheres to the guidelines provided above.
"""
