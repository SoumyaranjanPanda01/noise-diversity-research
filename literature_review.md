# Literature review and novelty assessment

Prepared 2026-11-20. Bibliographic details marked **[web-verified]** were checked in this session against the
publisher or the ACL Anthology / PMLR / NeurIPS / MLSys proceedings page (via web search results). **[repo-verified]**
entries were checked against the original PDFs earlier (see `docs/evidence_table.md`). **[verify]** marks details
taken from memory that should be checked by hand before submission (page numbers mainly). No finding below is
attributed to a paper beyond what its abstract or the evidence table supports.

## 1. Closest prior work and how this study differs

| work | setting | varies diversity at fixed corruption volume? | evaluates on held-out perturbation families? | relation to this paper |
|---|---|---|---|---|
| Alajrami et al., IJCNLP-AACL 2025 [repo-verified] | 7B to 72B LLMs; 6 instruction perturbations mixed; perturbed fraction swept at fixed size | No (every perturbed run mixes all 6) | No (same 6 at test) | closest; our design isolates diversity and holds out families |
| Li and Sen, arXiv 2026 [repo-verified] | GPT-2, Qwen2-0.5B, Llama-2-7B; label / typo / grammar noise, one type per run | No | No (clean test only) | single-type noise; no transfer test |
| Agrawal et al., ICLR 2025 workshop [repo-verified] | defences against perturbed instructions | No | No (training mixtures include test attacks) | adversarial, not held-out |
| Karpukhin et al., W-NUT 2019 [web-verified] | NMT; mixture of synthetic character noise; test on natural typos | No | Held-out *source* (natural vs synthetic), same broad family | shows a mixture of synthetic noise can transfer to natural typos; our families are more distinct |
| Kumar et al., W-NUT 2020 [web-verified] | BERT fine-tuned on misspelled text | No | No | noisy fine-tuning degrades BERT |
| Belinkov and Bisk, ICLR 2018 [repo-verified] | char-level NMT; synthetic and natural noise | No | Partly (cross-noise training/testing) | earliest text evidence that robustness is noise-specific |
| Rychalska et al., ICONIP 2019 (Springer LNCS 11955) [web-verified] | WildNLP corruption families on 4 tasks | No | evaluation framework; adversarial training on selected aspects | corruption-family taxonomy in NLP |
| Moradi and Samwald, EMNLP 2021 [web-verified] | BERT, RoBERTa, XLNet, ELMo; char/word perturbations | No (evaluation only) | n/a | sensitivity measurement |
| Gokhale et al., Findings ACL 2022 [web-verified] | data modification methods vs OOD generalisation and adversarial robustness | No | OOD datasets, not perturbation families | augmentation helps OOD and AR; filtering can hurt |
| Ng et al. (SSMBA), EMNLP 2020 [web-verified] | MLM-based augmentation for OOD robustness | No | OOD domains | augmentation for distribution shift |
| Hendrycks and Dietterich, ICLR 2019; AugMix, ICLR 2020 [repo-verified] | vision; held-out common corruptions; diverse augmentation | No | Yes (vision) | origin of held-out corruption evaluation |
| Mintun et al., NeurIPS 2021 [web-verified] | vision; augmentation-corruption similarity predicts gains; ImageNet-C-bar | partly | Yes, dissimilar corruptions | **strongest prior reason to expect our null**: gains shrink on dissimilar corruptions |
| Kang et al., arXiv 2019 [repo-verified] | vision; adversarial robustness transfer grid | No | Yes | robustness rarely transfers between types |
| Gulrajani and Lopez-Paz, ICLR 2021 [web-verified] | DomainBed; leave-one-domain-out | n/a | held-out domains | negative result: tuned ERM matches DG methods; methodological analogue |
| Jia and Liang, EMNLP 2017 [web-verified] | appended distracting sentences in SQuAD paragraphs | n/a | n/a | classic distractor stress test (they perturb the passage; we perturb the question) |
| Shi et al., ICML 2023 (PMLR 202) [web-verified] | GSM-IC; irrelevant context in math problems | n/a | n/a | LLM distractibility |
| Zhao et al., ICML 2021 (PMLR 139) [web-verified] | answer bias of LMs; contextual calibration | n/a | n/a | lens for the "No" shift; motivates the re-threshold diagnostic |
| NEFTune (Jain et al., ICLR 2024) [repo-verified] | embedding noise for instruction tuning | n/a | not evaluated on perturbed inputs | baseline |
| EDA (Wei and Zou, EMNLP-IJCNLP 2019) [repo-verified] | edit-based augmentation | n/a | n/a | baseline (made holdout-safe) |

Statistics and evaluation methodology:

| work | used for |
|---|---|
| Efron and Tibshirani 1993 (book) [repo-verified] | bootstrap |
| Koehn, EMNLP 2004 [verify pages 388-395] | paired bootstrap in NLP evaluation |
| Field and Welsh, JRSS-B 69(3):369-390, 2007 [web-verified] | cluster bootstrap |
| Dodge et al., arXiv:2002.06305, 2020 [web-verified] | seed (init and data order) variance in fine-tuning |
| Bouthillier et al., MLSys 2021 [web-verified] | accounting for multiple variance sources in benchmarks |
| Dror et al., ACL 2018, pp. 1383-1392 [web-verified] | significance testing practice in NLP |
| Lewis et al., EACL 2021, pp. 1000-1008 [web-verified] | train/test overlap inflates QA scores (cited in the audit; not in the 6-page paper for space) |
| Wang, Wang and Yang, NAACL 2022, pp. 4569-4586 [web-verified] | robustness survey |

Other entries in the revised bibliography: BoolQ (Clark et al., NAACL-HLT 2019) [verify pages 2924-2936], EDA
[verify pages 6382-6388], Qwen2.5 technical report (arXiv:2412.15115), LoRA (ICLR 2022), WordNet (CACM 1995).

## 2. What prior work established

1. Training on a perturbation type usually helps on that type, and robustness transfers poorly between dissimilar
   types (vision: Kang et al., Mintun et al.; text: Belinkov and Bisk).
2. Mixed noisy instruction tuning roughly preserves accuracy under the *same* perturbations (Alajrami et al.).
3. Mixtures of synthetic noise can help on natural noise of a similar kind (Karpukhin et al.).
4. LMs are sensitive to irrelevant context and carry answer biases that calibration partly removes (Jia and
   Liang; Shi et al.; Zhao et al.).

## 3. What this paper investigates, and what is new

* **Investigates:** whether diversity itself, separated from corruption volume and from test-time exposure,
  improves robustness to *held-out perturbation families* when fine-tuning a small instruction-tuned LM with
  LoRA.
* **New, as far as the review found:** a fixed-volume, family-held-out, two-fold design for text, with a
  pre-registered decision rule and full per-item records; a quantified null with both test-item and seed
  uncertainty; a family transfer matrix with intervals; a dose-response for in-family distractor exposure
  (133 examples suffice in this setup); and the decomposition of the distractor drop into an answer-rate shift
  and a discrimination loss.
* **Not new:** that robustness is largely type-specific (expected from vision and from Belinkov and Bisk), and
  that LMs are distractible.
* **Claim to avoid:** "no prior work exists". The review covered the closest work, not the whole literature. The
  revised paper makes no priority claim and positions the result as a controlled test of a common assumption.

## 4. What the current experiments cannot establish

Effects at larger model scale; other tasks or answer formats; passage-level or natural noise; families with
more headroom; whether the distractor gain generalises to unseen distractor sentences; mechanisms.

## 5. Gaps noted but not filled

* IEEE Xplore publications: the targeted search found no IEEE paper that tests perturbation diversity with
  held-out families. Most closely related work is in ACL Anthology, NeurIPS, ICLR and ICML venues; the
  bibliography reflects that. A reviewer at an IEEE venue may expect one or two IEEE citations; if you know a
  relevant IEEE paper (for example from IEEE/ACM TASLP on noise robustness), add it only after reading it.
* Shi et al. 2023 was listed in the lab log as "author's decision after reading it"; it is now cited only for its
  abstract-level finding (LLMs are distracted by irrelevant context). Please read it before submission.

## Sources consulted in this session

* SSMBA: https://aclanthology.org/2020.emnlp-main.97
* Karpukhin et al.: https://aclanthology.org/D19-5506/
* Kumar et al.: https://aclanthology.org/2020.wnut-1.3
* Moradi and Samwald: https://aclanthology.org/2021.emnlp-main.117
* Jia and Liang: https://aclanthology.org/D17-1215/
* Gokhale et al.: https://aclanthology.org/2022.findings-acl.213
* Wang et al. survey: https://preview.aclanthology.org/setup/2022.naacl-main.339
* Dror et al.: https://aclanthology.org/P18-1128/
* Lewis et al.: https://aclanthology.org/2021.eacl-main.86
* Shi et al.: https://proceedings.mlr.press/v202/shi23a.html
* Zhao et al.: https://proceedings.mlr.press/v139/zhao21c.html
* Mintun et al.: https://papers.nips.cc/paper_files/paper/2021/file/1d49780520898fe37f0cd6b41c5311bf-Paper.pdf
* Bouthillier et al.: https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html
* Gulrajani and Lopez-Paz: https://iclr.cc/virtual/2021/poster/2998
* Dodge et al.: https://arxiv.org/abs/2002.06305
* Field and Welsh: https://ideas.repec.org/a/bla/jorssb/v69y2007i3p369-390.html
* Rychalska et al.: https://researchr.org/publication/RychalskaBGB19
