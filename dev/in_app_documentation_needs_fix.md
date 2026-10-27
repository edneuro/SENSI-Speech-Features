
# In app documentation - Issues to fix

Look, I told you there were inconsistencies in the app documentation. I guess you tried to fix the wrong links and some content, but it was not good enough. there is many issues.

- Some links go to different papers than what is being mentioned/referenced (e.g., some arxiv links). I thought you said you fixed it, but you did not.
- There is incomplete parts that we have not tackle, even though it should have been
- there is inconsistencies in descriptions.
- Some claims you made reference some papers, but those papers do not support the claim
- You have to make sure the claims you make are backed up by the actual content of the paper, inference is good, but you cant just assume your training knowledge is good enough, you have to fact check. remember this is a tool that scientist will use. Citations have to be correct too, dont just guess, fact check it too.

Below are some issues I found while doing a quick pass.
## Under Word level features

### Missing or need fix

#### Structural identifier columns
- doc_id and paragraph_id are hardcoded to 1 in the extractor — multi-document and paragraph-aware processing aren't part of the current pipeline. 
	- doc_id: "Document. Currently always 1 — single-document pipeline. Placeholder for future multi-stimulus runs." this is something to fix, cause we may not do only one document at a time, but multiple ones.
	- paragraph_id: "Paragraph within the document. Currently always 1 — paragraph segmentation is not yet implemented; reserved for future use." This is something to fix, cause a document may have many paragraphs.

- Easy to confuse with Onset / End (capitalized), which are the acoustic word boundaries in seconds from extractors/aligner.py. These character offsets are lowercase start / end and are in characters within the sentence, not seconds.
	- will this be an issue with accessing data from the dataframe?
#### Lexical-Linguistic
- Dependency parse:
	- deps: "Enhanced-UD parallel dependencies. Currently always "NA" — spaCy's en_core_web_sm does not emit enhanced deps; reserved for future tagger upgrades."
	- misc: (UD MISC field. Currently used only for SpaceAfter=No (token is not followed by whitespace, e.g. before punctuation in "cat."). Otherwise "NA".)
#### Predictability
- gpt-2 surprisal: in the md file you mentioned that the limit is 1024 tokens and we use a 512 token window for longer texts. during brainstorming you told me the max content windows for gpt-2 was 512. so if it is 1024, why are we not using a 1024 sliding window? I thought we were calculating the surprisal per word and doing the slide, but now you mention something about a stride. are you calculating many surprisals at once? I am confused.
- in gpt-2 surprisal, this link is wrong: 
	Goodkind 2018 Goodkind & Bicknell. Predictive power of word surprisal for reading times is a linear function of language model quality. CMCL 2018. arXiv:1804.10742 — better LMs give better surprisal estimates for reading time.
- 
### Inconsistencies 

#### Word Frequency and Surprisal

there seems to be an inconsistency on the claims we are making

Inside word frequency md file:
	I went to the "Brysbaert 2018" paper and asked  comet if the following statement was true.
	"The Zipf scale is preferred over raw counts or log-frequency because it is linear in perceptual difficulty: a 1-unit drop in Zipf produces a roughly constant increase in reading time / N400 amplitude across the range ([Brysbaert 2018](#brysbaert-2018))."
	It said: No. That exact statement is not correct for this paper.

Inside gpt 2 surprisal md file:
	Question: Is this true for the following papers? Surprisal Theory (Hale 2001, Levy 2008) proposes that processing cost is linear in surprisal: doubling improbability doubles the cognitive cost of integrating a word into the unfolding context. The N400 has been shown to track surprisal more tightly than raw probability (Wlotko 2012).
	Sonnet:
	"""Based on the documents, I can confirm the following:
	Surprisal Theory and Linear Relationship
	**Partially accurate.** The papers do propose surprisal theory, but the characterization needs refinement:
	- **Hale (2001)** proposes that cognitive load is measured by "surprisal" - the logarithm of the word's probability given its prefix. The paper states that surprisal is defined by taking the log of the combined difficulty measure, creating a "word-based measure of cognitive effort."
	- **Levy (2008)** explicitly states that "surprisal serves as a causal bottleneck between the linguistic representations constructed during sentence comprehension and the processing difficulty incurred at a given word." The paper notes that "the theory proposes that conditional probability affects difficulty in a **log scale**. That is, the ratio rather than the difference between the conditional (or Cloze) probabilities should be the determinant of differential" difficulty.
    So while surprisal theory does propose a relationship between surprisal and processing cost, the "doubling improbability doubles cognitive cost" characterization is somewhat misleading. Since surprisal is already on a log scale (negative log probability), the relationship between surprisal itself and processing cost is proposed to be direct/linear, but the relationship between raw probability and cost is logarithmic.
    N400 and Surprisal vs. Raw Probability
    **Cannot confirm from these papers.** The third paper (1-s2.0-S1053811912004508-main.pdf) discusses the N400 extensively and mentions that "N400 amplitude was graded with expectancy, as amplitudes decreased with increasing constraint." However, I don't see a specific claim in any of these papers that the N400 tracks surprisal more tightly than raw probability, nor is there a "Wlotko 2012" citation in the provided documents."""