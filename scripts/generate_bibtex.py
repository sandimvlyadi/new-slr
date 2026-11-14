import csv
import os
import re

os.makedirs("/data/Smart-X/new-slr/thesis", exist_ok=True)

scopus_by_doi = {}
scopus_by_title = {}
with open("/data/Smart-X/new-slr/search/scopus_export_2026-11-10.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["DOI"]:
            scopus_by_doi[r["DOI"].strip().lower()] = r
        scopus_by_title[r["Title"].strip().lower()[:60]] = r

pubmed_by_doi = {}
with open("/data/Smart-X/new-slr/search/pubmed_records.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["doi"]:
            pubmed_by_doi[r["doi"].strip().lower()] = r

with open("/data/Smart-X/new-slr/screening/02_fulltext_candidates.csv") as f:
    cands = {r["id"]: r for r in csv.DictReader(f)}

with open("/data/Smart-X/new-slr/verification/03_fulltext_criteria_verification.csv") as f:
    ver = {r["id"]: r for r in csv.DictReader(f)}

manual_authors = {
    "M303": "Patel, Priyanshi Mehulbhai and Bairy, G. Muralidhar",
    "M305": "Aktemur, Ilknur and Oksuz, Ilkay",
    "M306": "Felfeliyan, Banafshe and Zhou, Yuyue and Ghosh, Shrimanti and Liu, Shaobo and Hareendranathan, Abhilash and Jaremko, Jacob L.",
    "M329": "Fu, Meiling and Zhang, Qinghua and Deng, Ruhan and Yang, Ying",
    "M335": "Konwer, Aishik and Xu, Xuan and Bae, Joseph and Chen, Chao and Prasanna, Prateek",
    "C007": "Tan, Yan Ming and Chong, Raphael Quek Hao and Hargreaves, Carol Anne",
    "C015": "Bruntha, S. and Chidambaram, M.",
    "C037": "Prema, G. and Rithanya, P. and Shiva Priya, A.",
    "C093": "Coskun, Huseyin and Katilmis, Zekeriya",
}

def format_scopus_authors(raw_full, raw_short):
    if raw_full:
        parts = [re.sub(r"\s*\(\d+\)", "", p).strip() for p in raw_full.split(";") if p.strip()]
        if parts:
            return " and ".join(parts)
    if raw_short:
        parts = [p.strip() for p in raw_short.split(";") if p.strip()]
        if parts:
            return " and ".join(parts)
    return None

entries = []
for pid, c in sorted(cands.items()):
    doi = (c["doi"] or "").strip()
    sc = scopus_by_doi.get(doi.lower()) or scopus_by_title.get(c["title"].strip().lower()[:60])
    pm = pubmed_by_doi.get(doi.lower())
    authors = None
    if sc:
        authors = format_scopus_authors(sc.get("Author full names", ""), sc.get("Authors", ""))
    if not authors and pm and pm.get("authors"):
        parts = [p.strip() for p in pm["authors"].split(";") if p.strip()]
        authors = " and ".join(parts)
    if not authors:
        authors = manual_authors.get(pid, "Unknown")
    year = c["year"]
    title = c["title"].strip()
    if sc and sc.get("Title"):
        title = sc["Title"].strip()
    venue = (sc["Source title"].strip() if sc and sc.get("Source title") else c["venue"].strip())
    vol = sc["Volume"].strip() if sc and sc.get("Volume") else ""
    issue = sc["Issue"].strip() if sc and sc.get("Issue") else ""
    pstart = sc["Page start"].strip() if sc and sc.get("Page start") else ""
    pend = sc["Page end"].strip() if sc and sc.get("Page end") else ""
    artno = sc["Art. No."].strip() if sc and sc.get("Art. No.") else ""
    pages = f"{pstart}--{pend}" if pstart and pend else (pstart or artno)
    is_conf = any(w in venue.lower() for w in ["conference", "symposium", "workshop", "proceedings", "lecture notes", "communications in computer"])
    btype = "inproceedings" if is_conf else "article"
    jfield = "booktitle" if is_conf else "journal"
    if pid in ver:
        dec = ver[pid]["reviewer_ft_decision"]
        status_note = f"SLR-M ID: {pid} ({dec})"
    else:
        status_note = f"SLR-M ID: {pid} (abstract-only / unretrieved)"
    lines = [
        f"@{btype}{{{pid},",
        f"  author = {{{authors}}},",
        f"  title = {{{{{title}}}}},",
        f"  {jfield} = {{{venue}}},",
        f"  year = {{{year}}},",
    ]
    if vol:
        lines.append(f"  volume = {{{vol}}},")
    if issue:
        lines.append(f"  number = {{{issue}}},")
    if pages:
        lines.append(f"  pages = {{{pages}}},")
    if doi:
        lines.append(f"  doi = {{{doi}}},")
    lines.append(f"  note = {{{status_note}}}")
    lines.append("}")
    entries.append("\n".join(lines))

foundational = """
@article{vanderHeijde2000SvdH,
  author = {van der Heijde, Desiree},
  title = {{How to read radiographs according to the Sharp/van der Heijde method}},
  journal = {The Journal of Rheumatology},
  year = {2000},
  volume = {27},
  number = {1},
  pages = {261--263}
}

@inproceedings{Yang2026RAMH1200,
  author = {Yang, Songxiao and Wang, Haolin and Fu, Yao and Peng, Junmu and Fan, Lin and Chen, Hongruixuan and Song, Jian and Ikebe, Masayuki and Takamaeda-Yamazaki, Shinya and Okutomi, Masatoshi and Kamishima, Tamotsu and Ou, Yafei},
  title = {{RAM-H1200: A Unified Evaluation and Dataset on Hand Radiographs for Rheumatoid Arthritis}},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS Datasets and Benchmarks Track)},
  year = {2026},
  note = {arXiv:2605.05616}
}

@inproceedings{Stolpovsky2023RheumaVIT,
  author = {Stolpovsky, Alexander and Dakhova, Elizaveta and Druzhinina, Polina and Postnikova, Polina and Kudinsky, Daniil and Smirnov, Alexander and Sukhinina, Anastasia and Lila, Alexander and Kurmukov, Anvar},
  title = {{RheumaVIT: Transformer-Based Model for Automated Scoring of Hand Joints in Rheumatoid Arthritis}},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year = {2023},
  pages = {2578--2586},
  doi = {10.1109/ICCVW60793.2023.00272}
}

@inproceedings{Caron2021DINO,
  author = {Caron, Mathilde and Touvron, Hugo and Misra, Ishan and Jegou, Herve and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  title = {{Emerging Properties in Self-Supervised Vision Transformers}},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2021},
  pages = {9650--9660}
}

@inproceedings{He2022MAE,
  author = {He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Dollar, Piotr and Girshick, Ross},
  title = {{Masked Autoencoders Are Scalable Vision Learners}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022},
  pages = {16000--16009}
}

@inproceedings{Khosla2020SupCon,
  author = {Khosla, Prannay and Teterwak, Piotr and Wang, Chen and Sarna, Aaron and Tian, Yonglong and Isola, Phillip and Maschinot, Aaron and Liu, Ce and Krishnan, Dilip},
  title = {{Supervised Contrastive Learning}},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2020},
  volume = {33},
  pages = {18661--18673}
}

@inproceedings{Lin2017Focal,
  author = {Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Dollar, Piotr},
  title = {{Focal Loss for Dense Object Detection}},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year = {2017},
  pages = {2980--2988}
}

@article{Cao2020CORAL,
  author = {Cao, Wenzhi and Mirjalili, Vahid and Raschka, Sebastian},
  title = {{Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation}},
  journal = {Pattern Recognition Letters},
  year = {2020},
  volume = {140},
  pages = {325--331},
  doi = {10.1016/j.patrec.2020.11.008}
}

@article{Shi2023CORN,
  author = {Shi, Xintong and Cao, Wenzhi and Raschka, Sebastian},
  title = {{Deep Neural Networks for Rank-Consistent Ordinal Regression Based on Conditional Probabilities}},
  journal = {Pattern Analysis and Applications},
  year = {2023},
  volume = {26},
  pages = {941--955},
  doi = {10.1007/s10044-023-01181-9}
}

@inproceedings{Menon2021LogitAdjustment,
  author = {Menon, Aditya Krishna and Jayasumana, Sadeep and Rawat, Ankit Singh and Jain, Himanshu and Veit, Andreas and Kumar, Sanjiv},
  title = {{Long-Tail Learning via Logit Adjustment}},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2021}
}

@inproceedings{Ren2020BalancedSoftmax,
  author = {Ren, Jiawei and Yu, Cunjun and Ma, Xiao and Zhao, Haiyu and Yi, Shuai and Li, Hongsheng},
  title = {{Balanced Meta-Softmax for Long-Tailed Visual Recognition}},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2020},
  volume = {33},
  pages = {4175--4186}
}

@inproceedings{Kang2020Decoupling,
  author = {Kang, Bingyi and Xie, Saining and Rohrbach, Marcus and Yan, Zhicheng and Gordo, Albert and Feng, Jiashi and Kalantidis, Yannis},
  title = {{Decoupling Representation and Classifier for Long-Tailed Recognition}},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2020}
}

@inproceedings{Angelopoulos2021RAPS,
  author = {Angelopoulos, Anastasios Nikolas and Bates, Stephen and Malik, Jitendra and Jordan, Michael I.},
  title = {{Uncertainty Sets for Image Classifiers using Conformal Prediction}},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2021}
}

@inproceedings{Kull2019Dirichlet,
  author = {Kull, Meelis and Perello Nieto, Miquel and Kangsepp, Markus and Silva Filho, Telmo and Song, Hao and Flach, Peter},
  title = {{Beyond Temperature Scaling: Obtaining Well-Calibrated Multi-Class Probabilities with Dirichlet Calibration}},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2019},
  volume = {32}
}

@article{Northcutt2021ConfidentLearning,
  author = {Northcutt, Curtis G. and Jiang, Lu and Chuang, Isaac L.},
  title = {{Confident Learning: Estimating Uncertainty in Dataset Labels}},
  journal = {Journal of Artificial Intelligence Research},
  year = {2021},
  volume = {70},
  pages = {1373--1411},
  doi = {10.1613/jair.1.12125}
}

@inproceedings{Tarvainen2017MeanTeacher,
  author = {Tarvainen, Antti and Valpola, Harri},
  title = {{Mean Teachers are Better Role Models: Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results}},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2017},
  volume = {30}
}

@inproceedings{Schneider2020TTABN,
  author = {Schneider, Steffen and Rusak, Evgenia and Eck, Luisa and Bringmann, Oliver and Brendel, Wieland and Bethge, Matthias},
  title = {{Improving Robustness Against Common Corruptions by Covariate Shift Adaptation}},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2020},
  volume = {33},
  pages = {11539--11551}
}

@inproceedings{Wang2021Tent,
  author = {Wang, Dequan and Shelhamer, Evan and Liu, Shaoteng and Olshausen, Bruno and Darrell, Trevor},
  title = {{Tent: Fully Test-Time Adaptation by Entropy Minimization}},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2021}
}

@article{Page2021PRISMA,
  author = {Page, Matthew J. and McKenzie, Joanne E. and Bossuyt, Patrick M. and Boutron, Isabelle and Hoffmann, Tammy C. and Mulrow, Cynthia D. and Shamseer, Larissa and Tetzlaff, Jennifer M. and Akl, Elie A. and Brennan, Sue E. and others},
  title = {{The PRISMA 2020 statement: an updated guideline for reporting systematic reviews}},
  journal = {BMJ},
  year = {2021},
  volume = {372},
  pages = {n71},
  doi = {10.1136/bmj.n71}
}

@article{Campbell2020SWiM,
  author = {Campbell, Mhairi and McKenzie, Joanne E. and Sowden, Amanda and Katikireddi, Srinivasa Vittal and Brennan, Sue E. and Ellis, Simon and Hartmann-Boyce, Jamie and Ryan, Rebecca and Shepperd, Sasha and Thomas, James and others},
  title = {{Synthesis without meta-analysis (SWiM) in systematic reviews: reporting guideline}},
  journal = {BMJ},
  year = {2020},
  volume = {368},
  pages = {l6890},
  doi = {10.1136/bmj.l6890}
}

@article{Chen2019KneeOA,
  author = {Chen, Pingjun and Gao, Linlin and Shi, Xiaoshuang and Allen, Kyle and Yang, Lin},
  title = {{Fully automatic knee osteoarthritis severity grading using deep neural networks with a novel ordinal loss}},
  journal = {Computerized Medical Imaging and Graphics},
  year = {2019},
  volume = {75},
  pages = {84--92},
  doi = {10.1016/j.compmedimag.2019.06.002}
}

@inproceedings{Gal2016Dropout,
  author = {Gal, Yarin and Ghahramani, Zoubin},
  title = {{Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning}},
  booktitle = {Proceedings of the 33rd International Conference on Machine Learning (ICML)},
  year = {2016},
  volume = {48},
  pages = {1050--1059}
}

@inproceedings{Guo2017Calibration,
  author = {Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  title = {{On Calibration of Modern Neural Networks}},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning (ICML)},
  year = {2017},
  volume = {70},
  pages = {1321--1330}
}

@inproceedings{Sensoy2018Evidential,
  author = {Sensoy, Murat and Kaplan, Lance and Kandemir, Melih},
  title = {{Evidential Deep Learning to Quantify Classification Uncertainty}},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2018},
  volume = {31}
}

@inproceedings{Romano2020APS,
  author = {Romano, Yaniv and Sesia, Matteo and Candes, Emmanuel},
  title = {{Classification with Valid and Adaptive Coverage}},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2020},
  volume = {33},
  pages = {3581--3591}
}

@book{Vovk2005Conformal,
  author = {Vovk, Vladimir and Gammerman, Alex and Shafer, Glenn},
  title = {{Algorithmic Learning in a Random World}},
  publisher = {Springer},
  year = {2005},
  doi = {10.1007/b106715}
}

@inproceedings{Kumar2022LPFT,
  author = {Kumar, Ananya and Raghunathan, Aditi and Jones, Robbie and Ma, Tengyu and Liang, Percy},
  title = {{Fine-Tuning Can Distort Pretrained Features and Underperform Out-of-Distribution}},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2022}
}

@article{Gretton2012MMD,
  author = {Gretton, Arthur and Borgwardt, Karsten M. and Rasch, Malte J. and Scholkopf, Bernhard and Smola, Alexander},
  title = {{A Kernel Two-Sample Test}},
  journal = {Journal of Machine Learning Research},
  year = {2012},
  volume = {13},
  pages = {723--773}
}

@article{Halabi2019RSNA,
  author = {Halabi, Safwan S. and Prevedello, Luciano M. and Kalpathy-Cramer, Jayashree and Mamonov, Artem B. and Bilbily, Alexander and Cicero, Mark and Pan, Ian and Pereira, Lucas Araujo and Sousa, Rafael Teixeira and Abdala, Nitamar and others},
  title = {{The RSNA Pediatric Bone Age Machine Learning Challenge}},
  journal = {Radiology},
  year = {2019},
  volume = {290},
  number = {2},
  pages = {498--503},
  doi = {10.1148/radiol.2018180736}
}

@article{Holm1979,
  author = {Holm, Sture},
  title = {{A simple sequentially rejective multiple test procedure}},
  journal = {Scandinavian Journal of Statistics},
  year = {1979},
  volume = {6},
  number = {2},
  pages = {65--70}
}

@inproceedings{Howard2018ULMFiT,
  author = {Howard, Jeremy and Ruder, Sebastian},
  title = {{Universal Language Model Fine-tuning for Text Classification}},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year = {2018},
  pages = {328--339}
}

@article{Hinton2015Distillation,
  author = {Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  title = {{Distilling the Knowledge in a Neural Network}},
  journal = {arXiv preprint arXiv:1503.02531},
  year = {2015}
}

@inproceedings{Szegedy2016LabelSmoothing,
  author = {Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
  title = {{Rethinking the Inception Architecture for Computer Vision}},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2016},
  pages = {2818--2826}
}

@inproceedings{Diaz2019SORD,
  author = {Diaz, Raul and Marathe, Amit},
  title = {{Soft Labels for Ordinal Regression}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019},
  pages = {4738--4747}
}

@book{Efron1993Bootstrap,
  author = {Efron, Bradley and Tibshirani, Robert J.},
  title = {{An Introduction to the Bootstrap}},
  publisher = {Chapman and Hall/CRC},
  year = {1993}
}

@article{Cohen1968WeightedKappa,
  author = {Cohen, Jacob},
  title = {{Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit}},
  journal = {Psychological Bulletin},
  year = {1968},
  volume = {70},
  number = {4},
  pages = {213--220}
}

@inproceedings{Geifman2017Selective,
  author = {Geifman, Yonatan and El-Yaniv, Ran},
  title = {{Selective Classification for Deep Neural Networks}},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2017},
  volume = {30}
}

@article{C047,
  author = {Anonymous},
  title = {{From Kellgren-Lawrence to Calcium Pyrophosphate Crystal Deposition: A Soft-Labelling Framework for Knee Osteoarthritis Assessment}},
  journal = {arXiv preprint arXiv:2605.28176},
  year = {2026},
  doi = {10.48550/arxiv.2605.28176},
  note = {SLR-M ID: C047 (methodological reference outside PRISMA: 4 ordinal soft-label distribution formulations)}
}
"""

with open("/data/Smart-X/new-slr/thesis/references.bib", "w") as f:
    f.write("% ==========================================================================\n")
    f.write("% Daftar Pustaka Terverifikasi untuk Bab I, Bab II & Bab III Tesis (SLR-M)\n")
    f.write("% ==========================================================================\n\n")
    f.write(foundational.strip() + "\n\n")
    f.write("\n\n".join(entries) + "\n")

print("Wrote", len(entries), "SLR-M entries + 38 foundational entries to thesis/references.bib")
