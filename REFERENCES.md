# References

Short keys in `[brackets]` are used in code comments (`focus_core/constants.py` and the three main scripts).
Each entry says what it is used for and what could / could not be verified when it was looked up (abstract or listing page only,
unless stated). **Please check volume/page details before citing any of these in a paper.**

## EEG patterns (focused vs. mind-wandering)
- **[Kam2022]** Kam JWY, Rahnuma T, Park YE, Hart CM (2022). Electrophysiological markers of mind wandering: A systematic review.
  *NeuroImage* 258. PMID 35700946. Spectral findings are less robust than ERP findings but point to more delta/theta/alpha and less beta
  during mind wandering than on-task; variability depends on task context.
- **[Groot2021]** Groot JM, Boayue NM, Csifcsak G, Boekel W, et al. (2021). Probing the neural signature of mind wandering with simultaneous
  fMRI-EEG and pupillometry. *NeuroImage* 224:117412. Task-unrelated thought: widespread alpha/theta/delta increase but not beta; larger
  baseline pupil size.
- **[Jin2019]** Jin CY, Borst JP, van Vugt MK (2019). Predicting task-general mind-wandering with EEG. PMID 30850931 (PMC6711882). Alpha
  power was the most predictive EEG marker; accuracy 0.50-0.85. *(Journal name not shown in the retrieved text.)*
- **[Arnau2019]** Arnau S. The electrophysiological signature of mind wandering. *bioRxiv* doi:10.1101/819805 (preprint; cited as Arnau et
  al., 2020 in a later paper). Alpha power higher on trials with reported mind wandering.
- **[vanSon2019]** van Son D, De Blasio FM, Fogarty JS, Angelidis A, Barry RJ, Putman P (2019). Frontal EEG theta/beta ratio during mind
  wandering episodes. *Biological Psychology* 140:19-27. Frontal TBR higher during mind wandering (n = 26); no link to trait attentional control.
- **[Frontiers2025]** *Frontiers in Neuroscience* (2025), article 1654827, doi:10.3389/fnins.2025.1654827. *(Authors/title not retrieved.)*
  Mind wandering showed more 8-10 Hz alpha over frontal-posterior regions.
- **[bioRxiv2023]** "Distinct electrophysiological signatures of intentional and unintentional mind-wandering revealed by low-frequency EEG
  markers", *bioRxiv* doi:10.1101/2023.03.21.533634. *(Authors not retrieved.)* Greater alpha in task-unrelated thought in many studies (some
  contradict); frontal theta also increases with externally directed attention demands.
- Not used: the engagement index beta/(alpha+theta) -- a vigilance-decrement study (authors not retrieved) found it and frontal theta did not
  show systematic effects.

## Camera eye features
- **[VanOrden2000]** Van Orden KF, Jung T-P, Makeig S (2000). Combined eye activity measures accurately estimate changes in sustained visual
  task performance. *Biological Psychology* 52:221-240.
- **[Schleicher2008]** Schleicher R, Galley N, Briest S, Galley L (2008). Blinks and saccades as indicators of fatigue in sleepiness warnings:
  looking tired? *Ergonomics*. doi:10.1080/00140130701817062. 129 participants; blink duration, lid-reopening delay, blink interval and lid
  closure speed best indicators; substantial inter-individual differences.
- **[FAA1994]** FAA Office of Aerospace Medicine technical reports (1994, 1996), "Blinks, saccades, and fixation pauses during vigilance task
  performance". *(Authors not retrieved.)* Time-on-task effects on blink and saccade measures.
- **[ZhengLu2017]** Zheng W-L, Lu B-L (2017). A multimodal approach to estimating vigilance using EEG and forehead EOG. arXiv:1611.08492
  (SEED-VIG dataset: simulated driving, forehead EOG + EEG, PERCLOS labels). *(Journal: J Neural Eng, from memory.)*
- **[Soukupova2016]** Soukupova T, Cech J (2016). Real-time eye blink detection using facial landmarks. 21st Computer Vision Winter Workshop,
  Rimske Toplice, Slovenia. Eye aspect ratio from facial landmarks; blinks from its temporal pattern.
- **[GarciaAgundez2018]** Garcia-Agundez A, Ochs T, Konrad R, Caserman P, Gobel S (2018). Eye aspect ratio based blink rate detection and its
  potential use for Parkinson's disease. IWBBIO 2018. Webcam EAR blink detection: 82% precision / 75% recall in 10 healthy users.
- **[Danisman2010]** Danisman T, Bilasco IM, Djeraba C, Ihaddadene N (2010). Drowsy driver detection system using eye blink patterns. ICMWI 2010,
  230-233. doi:10.1109/ICMWI.2010.5648121. Webcam blink detection, 94% accuracy on the JZU database.
- **[AttentionLapse2023]** arXiv:2304.10588, "Detecting worker attention lapses in human-robot interaction: an eye tracking and multimodal
  sensing study". *(Authors not retrieved; secondary source.)* Long fixation, looking away, slow/irregular saccades, eye closure, high blink
  rate as lapse patterns; a person can mind-wander while fixating correctly.

## Training task: pursuit, lapses, clicks
- **[Fransson2008]** Fransson P-A, Patel M, Magnusson M (2008). Effects of 24-hour and 36-hour sleep deprivation on smooth pursuit and
  saccadic eye movements. *J Vestib Res* 18(4). doi:10.3233/VES-2008-18404.
- **[Frontiers2022]** "Differential saccade-pursuit coordination under sleep loss and low-dose alcohol", *Frontiers in Neuroscience* (2022),
  doi:10.3389/fnins.2022.1067722. *(Authors not retrieved.)* Pursuit-gain loss under sleep loss and low-dose alcohol; caffeine reduced the
  pursuit deficit after sleep loss.
- **[Peiris2006]** Peiris et al. (2006). Lapses of responsiveness during an extended visuomotor tracking task in non-sleep-deprived subjects.
  *J Sleep Res* 15:291-300. *(Full author list and exact title wording not retrieved.)*
- **[Jones2010]** Jones et al. (2010). Lapses of responsiveness: characteristics, detection, and underlying mechanisms. *Proc IEEE EMBS Annual
  Int Conf.* *(Authors/full title not retrieved.)* Lapses ~0.5-15 s; flat/incoherent tracking > 500 ms with eye closure = microsleep, without
  = attention lapse.
- **[PVT2020]** "Response speed measurements on the psychomotor vigilance test: how precise is precise enough?" *Sleep* (2020), PMID 32556295.
  *(Authors not retrieved.)* Lapses = RT >= 500 ms; timing bias should stay within about +-5 ms (SD <= 10 ms).
- **[QbTest]** QbTech, "Attention & impulse control parameters" (commercial documentation), qbtech.com/qbuser/interpretation/q5/. RT variation is
  described as the most sensitive attention parameter; responses < 150 ms are anticipatory. *(Not peer reviewed.)*
- **[gradCPT2013]** Gradual-onset continuous performance task paper, *Attention, Perception & Psychophysics*, doi:10.3758/s13414-012-0413-x.
  *(Authors not retrieved.)* Commission errors and RT variability increased over a 12-minute task.
- **[SART1998]** "Circadian rhythm of sleepiness and vigilance studied in man" (1998), MRC Cognition and Brain Sciences Unit bibliography
  entry 6095. *(Authors not retrieved.)* Commission errors increased with sleepiness; fast responses were mostly errors.
- **[Shelat2024]** Shelat S, Schooler JW, Giesbrecht B (2024). Predicting attentional lapses using response time speed in continuous performance
  tasks. *Frontiers in Cognition* 3:1460349. doi:10.3389/fcogn.2024.1460349.

## Context (complex score)
- **[Heishman2010]** Heishman SJ, Kleykamp BA, Singleton EG (2010). Meta-analysis of the acute effects of nicotine and smoking on human
  performance. *Psychopharmacology*. *(Volume/pages not retrieved.)*
- **[Myers2013]** Myers CS, Taylor RC, Salmeron BJ, Waters AJ, Heishman SJ (2013). Nicotine enhances alerting, but not executive, attention in
  smokers and nonsmokers. *Nicotine & Tobacco Research*.
- **[Mancuso2001]** Mancuso G, Lejeune M, Ansseau M (2001). Cigarette smoking and attention: processing speed or specific effects?
  *Psychopharmacology* 155(4).
- **[Caffeine60mg]** "Effects of a single, oral 60 mg caffeine dose on attention in healthy adult subjects". *(Authors/journal not retrieved.)*
  82 adults aged 40-60: improved sustained attention, peak saccadic velocity and reaction time.
- **[Kessler2005]** Kessler RC, Adler L, Ames M, et al. (2005). The World Health Organization Adult ADHD Self-Report Scale (ASRS). *Psychological
  Medicine* 35:245-256. **(Cited from memory -- not retrieved; verify.)**
- Alcohol: the pursuit-gain loss in [Frontiers2022] is the only peer-reviewed support found; a thesis on alcohol and saccadic measures could not
  be read.

## Reference norms and quality
- **[vanDijk2022]** van Dijk H, van Wingen G, Denys D, Olbrich S, van Ruth R, Arns M (2022). The two decades brainclinics research archive for
  insights in neurophysiology (TDBRAIN) database. *Scientific Data* 9:333. (Credit required by its data-use agreement.)
- **[Arns2013]** Arns M, Conners CK, Kraemer HC (2013). A decade of EEG theta/beta ratio research in ADHD: a meta-analysis. *J Atten Disord*
  17(5):374-383.
- **[Casadei2022]** Casadei V, Ferrero R (2022). Online skin-electrode contact quality monitoring in wearable devices: an EEG application. I2MTC 2022.
- **[Sensors2019]** "Quality assessment of single-channel EEG for wearable devices", *Sensors* 19(3):601, doi:10.3390/s19030601. *(Authors not retrieved.)*
- **[PurdueISCAS]** Purdue SPARC lab, EEG noise paper (ISCAS; title not retrieved): more contact pressure and area reduce dry-electrode noise by up to 50 dB.

## Background (not used in any formula)
- Lenartowicz A, Loo SK (2014). Use of EEG to diagnose ADHD. *Curr Psychiatry Rep* 16(11):498.
- Strzelczyk et al. (2026). Theta-beta ratio in ADHD: multiverse analysis. *eLife* 15:RP111114. *(Title partly retrieved.)*
- Cortese S, et al. (2016). Neurofeedback for ADHD: meta-analysis of clinical and neuropsychological outcomes from randomized controlled trials.
  *JAACAP* 55(6):444-455. *(From an earlier search; verify.)*

## Relaxation / meditation score (relax_score.py) -- EEG
- **[Lomas2015]** Lomas T, Ivtzan I, Fu CHY (2015). A systematic review of the neurophysiology of mindfulness on EEG oscillations. *Neuroscience &
  Biobehavioral Reviews* (published online 2015-10-09). 56 papers, 1715 subjects. Mindfulness was most commonly associated with enhanced alpha and theta
  power compared with an eyes-closed resting state, though not uniformly; co-presence of alpha and theta may signify relaxed alertness.
- **[Cahn2006]** Cahn BR, Polich J (2006). Meditation states and traits: EEG, ERP, and neuroimaging studies. *Psychological Bulletin* 132(2):180-211.
  Overall EEG slowing; theta and alpha activation related to proficiency; effects variable.
- **[Meditation2018]** "Review of the neural oscillations underlying meditation", *Frontiers in Neuroscience* 12:178 (2018), doi:10.3389/fnins.2018.00178
  (Goodarzi A, Girgis F, et al.; full author list not retrieved). FA and OM meditation increase anterior theta; alpha rises in posterior regions in both.
- **[AlphaThetaDepth]** "Alpha and theta brain oscillations play a complementary role in inducing levels of altered consciousness during meditation
  practice". *(Authors/venue/year not retrieved.)* Alpha and frontal-midline theta are the bands most robustly associated with meditation; alpha
  followed meditation depth while theta deactivated as it deepened.

## Relaxed vs. drowsy
- **[Chokroverty]** Chokroverty S, *Sleep Disorders Medicine*, chapter 18 (retrieved copy: sleep.pitt.edu). Alpha cessation defines the wake-to-sleep
  transition; slow rolling eye movements are among the first signs of drowsiness.
- **[Magosso2007]** Magosso E, Ursino M, Provini F, Montagna P (2007). Wavelet analysis of electroencephalographic and electro-oculographic changes during
  the sleep onset period. *(Proceedings; venue not shown.)* 15 healthy subjects: slow eye movements preceded alpha blocking and theta intrusion by 10-20 min.
- **[Strauss2022]** Strauss M, Sitt JD, Naccache L, Raimondo F (2022). *NeuroImage* 251:119003 (title not retrieved). Drowsiness is a distinct period between
  wakefulness and sleep, marked by transient alpha/theta changes that track falling responsiveness.
- **[Patent5813993]** US patent 5,813,993, "Alertness and drowsiness detection and tracking system". *(Weak source.)* With eyes closed, sleepiness reduces alpha
  and increases theta; with eyes open, increased alpha/theta can accompany sleepiness.

## Heart-rate variability and slow breathing (planned watch feature)
- **[Laborde2017]** Laborde S, Mosley E, Thayer JF (2017). Heart rate variability and cardiac vagal tone in psychophysiological research - recommendations
  for experiment planning, data analysis, and data reporting. *Frontiers in Psychology* 8:213. **(Title/venue from memory; verify.)**
- **[TaskForce1996]** Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology (1996). Heart rate
  variability: standards of measurement, physiological interpretation and clinical use. **(From memory; verify.)**
- **[Shaffer2017]** Shaffer F, Ginsberg JP (2017). An overview of heart rate variability metrics and norms. *Frontiers in Public Health* 5:258. (Venue from memory.)
- **[Laborde2022]** Laborde S, Allen MS, Borges U, Dosseville F, Hosang TJ, Iskra M, Mosley E, Salvotti C, Spolverato L, Zammit N, Javelle F (2022). Effects of
  voluntary slow breathing on heart rate and heart rate variability: a systematic review and a meta-analysis. 223 studies. *(Journal not shown in the retrieved page.)*
- **[SevozCouche2022]** Sevoz-Couche C, Laborde S (2022). Heart rate variability and slow-paced breathing: when coherence meets resonance. *Neuroscience &
  Biobehavioral Reviews*. Breathing at the resonant frequency (~0.1 Hz) increases cardiac oscillations (vagally mediated HRV, baroreflex sensitivity).
- **[Lehrer2014]** Lehrer PM, Gevirtz R (2014). Heart rate variability biofeedback: how and why does it work? *Frontiers in Psychology*.
- **[Laborde2021]** Laborde S, Iskra M, et al. (2021). Slow-paced breathing: influence of inhalation/exhalation ratio and of respiratory pauses on cardiac vagal
  activity. *Sustainability* 13:7775. Above ~9 breaths/min cardiac vagal activity shows in HF HRV; below it shifts to LF HRV.
- **[WatchECG2024]** Computing in Cardiology 2024, abstract 419 (authors not retrieved). Withings ScanWatch and Fitbit Sense single-lead (Lead I) 30-s ECGs vs
  a medical-grade ECG in 20 people: almost identical mean HR and HRV.
- **[Taoum2022]** Taoum A, Bisiaux A, Tilquin F, Le Guillou Y, et al. (2022). Validity of ultra-short-term HRV analysis using PPG - a preliminary study. *Sensors* 22(18):7047.
- **[PRVMeta]** "Accuracy of photoplethysmography-derived pulse rate variability compared with electrocardiography-derived heart rate variability: a systematic
  review and meta-analysis" (PMC13517957; authors not retrieved).
- **[JMIR2020]** "A novel smartphone app for the measurement of ultra-short-term and short-term heart rate variability: validity and reliability study",
  *JMIR mHealth and uHealth* 2020;8(7):e18761 (authors not retrieved). 1-minute recordings suggested for time-domain indices (SDNN, RMSSD, pNN50).
- **[SmartwatchHRV2019]** "Improving heart rate variability measurements from consumer smartwatches with machine learning", arXiv:1907.07496 (authors not
  retrieved). Reports RMSSD reliable on 10/30/60-s samples, unlike SDNN or frequency-domain measures.
