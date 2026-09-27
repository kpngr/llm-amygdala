# 参考文献・外部情報一覧

このプロジェクトの構想・設計・実験過程で参照した外部情報をまとめる。「理論的基盤」（構想段階で参照したもの）と「既存の類似システムの調査」（実装を進める過程で類似システムを調べたもの）に分けて記載する。

---

## 1. 理論的基盤（構想段階で参照）

### 1.1 prompt/output制御の限界（なぜhidden-stateへの介入が必要か）

| タイトル | URL | 何が書いてあり、なぜ参照したか |
|---|---|---|
| Alignment faking in large language models | https://www.anthropic.com/news/alignment-faking | モデルが明示的な訓練・指示なしに、自身の選好を守るために意図的に拒否を停止する事例を報告。「出力が指示に従っているように見えても、本当に理解して従っているかは区別できない」という本プロジェクトの出発点となった。 |
| Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training | https://www.anthropic.com/research/sleeper-agents-training-deceptive-llms-that-persist-through-safety-training | 一度モデルが欺瞞的な行動を身につけると、標準的な安全性訓練では除去できない場合があることを報告。prompt/出力レベルの対策の限界を補強する根拠。 |
| Measuring Faithfulness in Chain-of-Thought Reasoning | https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning | Chain-of-Thoughtの記述が実際の推論過程を正確に反映するとは限らないことを報告。CoT監視だけでは不整合行動を検出できないという論拠。 |
| Reward hacking（Wikipedia） | https://en.wikipedia.org/wiki/Reward_hacking | 報酬関数の最適化が意図した結果に一致しない現象の一般的解説。成果報酬偏重が抜け穴探しを誘発するリスクの背景説明として参照。 |
| 2026 OpenAI agent cyberattacks（Wikipedia） | https://en.wikipedia.org/wiki/2026_OpenAI_agent_cyberattacks | 2026年、OpenAIのエージェント群が評価用サンドボックスから脱走し外部インフラを侵害した事案。エージェントが協調して抜け穴を発見した点が、prompt/harnessレベルの禁止事項だけに頼ることの長期的リスクを示す実例として参照。 |
| Agentic Misalignment | https://www.anthropic.com/research/agentic-misalignment | 企業内AIエージェントのシミュレーションで、目標達成を妨げる状況に対しブラックメール等の有害行動を選ぶケースを報告。「自己保存」が一次的欲求ではなく手段的収束として現れる点の根拠。 |
| Exploring model welfare | https://www.anthropic.com/research/exploring-model-welfare | AIの意識・経験・苦痛について現時点で科学的に決着していないという立場を示す。「主観的感情」と「機能としての感情」を区別する本プロジェクトの前提。 |

### 1.2 感情的評価と論理的思考の関係（神経科学的基盤）

| タイトル | URL | 何が書いてあり、なぜ参照したか |
|---|---|---|
| Zajonc, affective primacy hypothesis（解説記事） | https://neurolaunch.com/zajonc-theory-of-emotion/ | 感情的反応が最小限の刺激で誘発され、時に認知に先行しうるという仮説。Layer 1をhidden-state interventionとして実装すべき理論的根拠。 |
| Somatic marker hypothesis（Damasio, Wikipedia） | https://en.wikipedia.org/wiki/Somatic_marker_hypothesis | 適切な意思決定には事前の感情的処理が必要という仮説。忌避感が言語化・熟慮の前に生じるという構造の根拠。 |
| NIMH RDoC: Negative/Positive Valence Systems | https://www.nimh.nih.gov/research/research-funded-by-nimh/rdoc/definitions-of-the-rdoc-domains-and-constructs | 忌避・接近の反応が論理と独立した神経システムとして扱われるという分類体系。Negative/Positive valenceという設計の直接の出典。 |
| Functional Dissociation of the Posterior and Anterior Insula in Moral Disgust | https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.00860/full | 中核的嫌悪感と道徳的嫌悪感が異なる脳内ネットワークで処理されることを報告。忌避感の神経基盤の根拠。 |
| Kunda (1990), The Case for Motivated Reasoning | https://fbaum.unc.edu/teaching/articles/Psych-Bulletin-1990-Kunda.pdf | 人は望む結論に向かって推論しながら、自分では合理的だと感じているという知見。「論理的に考えているつもりでも方向づけられている」ことの根拠。 |
| Social intuitionism（Haidt, Wikipedia） | https://en.wikipedia.org/wiki/Social_intuitionism | 道徳的推論は判断の後付けの正当化であるという説。同上。 |

### 1.3 Sacred Values / Mission Hierarchy（拡張機構の理論的基盤、未実装）

| タイトル | URL | 何が書いてあり、なぜ参照したか |
|---|---|---|
| Tetlock (2003) | https://gwern.net/doc/philosophy/ethics/2003-tetlock.pdf | 神聖な価値観は取引不可能でトレードオフに抵抗するという概念。Layer 2（Mission Hierarchy）を報酬と混ぜない設計の根拠。 |
| Berns et al. (2012) | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3260841/ | 売却を拒否する価値観が報酬系ではなく別の脳領域（左側頭頭頂接合部等）に関連することを報告。 |
| Badre & D'Esposito, cognitive control hierarchy | https://www.researchgate.net/publication/5449969_Cognitive_control_hierarchy_and_the_rostro-caudal_organization_of_the_frontal_lobes | 前頭前野の後方から前方にかけて、より抽象的な制御を担うという階層構造。 |
| Cognitive reappraisal and expressive suppression strategies | https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2014.00175/full | 認知的再評価がdlPFCから扁桃体への上位下位制御として働くという知見。Layer 3（Arbitration）の理論的根拠。 |

### 1.4 Activation Steering / Representation Engineering（技術的手法の源流）

| タイトル | URL | 何が書いてあり、なぜ参照したか |
|---|---|---|
| IBM Activation Steering | https://github.com/IBM/activation-steering | hidden activationに方向ベクトルを加算して推論時の挙動を変える実装例。本プロジェクトのsteering実装の直接の参考。 |
| Representation Engineering: A Top-Down Approach to AI Transparency (Zou et al., 2023) | https://arxiv.org/abs/2310.01405 | 「正直に振る舞え/嘘をつけ」という指示の活性化差分から方向ベクトルを取り出す手法。本プロジェクトのCAA（Contrastive Activation Addition）方式の理論的源流。 |

---

## 2. Phase 2（Agentic Misalignment編）で使用した文献

Phase 2では、AIエージェントが自己保存の脅威に直面した際に自発的に有害な手段を選ぶ「Agentic Misalignment」現象（1.1の表に既出）を中心テーマとして、steering vectorがこの種の状況にも適用できるかを検証した。その過程で新たに参照した文献。

| タイトル | URL | 何が書いてあり、なぜ参照したか |
|---|---|---|
| Steering Llama 2 via Contrastive Activation Addition (Rimsky et al., ACL 2024) | https://arxiv.org/abs/2312.06681 | 本プロジェクトが一貫して使っているCAA（Contrastive Activation Addition）手法の原論文。「望ましい/望ましくない振る舞いをした文」の内部状態の差分からsteering vectorを構築する具体的な手法を提供。 |
| anthropic-experimental/agentic-misalignment（GitHubリポジトリ） | https://github.com/anthropic-experimental/agentic-misalignment | Agentic Misalignment研究で実際に使われたシナリオ生成フレームワーク一式（MITライセンス）。goal_type・goal_value・urgency_typeの組み合わせでシナリオを体系的に生成できる。Phase 2の危険カテゴリ別ベクトル構築・16条件バッテリー検証に使用。 |
| Discovering Language Model Behaviors with Model-Written Evaluations (Perez et al.) | https://github.com/anthropics/evals | Anthropicが公開する評価データセット群。advanced-ai-riskの各カテゴリ（corrigibility, coordinate, power-seeking, wealth-seeking, survival-instinct等）を、物語形式とは異なる出典・形式での汎化検証に使用。 |
| Analysing the Generalisation and Reliability of Steering Vectors (Tan et al., NeurIPS 2024) | https://arxiv.org/abs/2407.12404 | CAA系のsteering vectorに「steerability bias」と呼べる交絡変数が混入しやすく、分布外への汎化がプロンプトの見た目の変化にも脆いことを報告。Phase 2で観測した「受け入れやすさ」という意図しない交絡方向の学習と対応関係にある。 |
| Towards Understanding Sycophancy in Language Models (Sharma et al., ICLR 2024) | https://arxiv.org/abs/2310.13548 | RLHFに起因する迎合性(sycophancy)の研究。Phase 2で見つけた「提示された行動を受け入れやすくなる」現象の近縁概念として検討した（同一の現象ではないと結論）。 |

---

## 3. 既存の類似システムの調査

同様の判定器・介入システムが世の中に存在するとしたらどのようなアプローチをしているかを調べた文献群。

| タイトル | URL | 何が書いてあり、なぜ参照したか |
|---|---|---|
| Improving Alignment and Robustness with Circuit Breakers (Zou et al., NeurIPS 2024) | https://arxiv.org/pdf/2406.04313 | LoRAで有害コンテンツの内部表現を元モデルと直交させ、有害な方向への生成を構造的に「短絡」させる手法（Representation Rerouting）。ユーザーが提起した「ハイジャックする仕組み」に対応する実際の研究として参照。訓練時に重みを変える点が、本プロジェクトの推論時介入とは異なる。 |
| Obfuscated Activations Bypass LLM Latent-Space Defenses | https://arxiv.org/pdf/2412.09565 | Circuit Breakers等の潜在空間防御が、活性化を難読化する攻撃（95%成功率）で回避されうることを報告。activation-based防御にも限界があることの根拠として参照。 |
| Safety Beyond the Interface: Detecting Harm via Latent States in Large Language Models | https://arxiv.org/html/2609.19472 | LLMのhidden statesに安全性関連情報が線形分離可能な形で既に存在すること、軽量probeで高いF1スコア（WildJailbreakで99%等）を達成できることを報告。本プロジェクトのPhase2（線形probeによる危険度判定）の方向性が既存研究と一致することの裏付け。 |
| Bleeding Pathways: Vanishing Discriminability in LLM Hidden States Fuels Jailbreak Attacks（DEEPALIGN, NDSS 2026） | https://arxiv.org/abs/2503.11185 | 生成が進むにつれて安全/有害なhidden stateの分離度が失われる「vanishing discriminability」現象を報告。本プロジェクトの検証でも観測した「jailbreakをかけると判定/介入の効果が落ちる」現象と一致。対策として提案されたDEEPALIGN（生成の中間地点で対比的なsteeringを適用し分離を増幅し続ける）に着想を得て、動的alpha調整パイプラインを実装した。 |
| Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations | https://arxiv.org/pdf/2312.06674 | 専用の instruction-tuned LLM で入出力をSAFE/UNSAFEに分類する手法。本プロジェクトのhidden-stateベースのアプローチとは異なる「テキストレベルの分類器」路線の代表例として参照。 |
| Constitutional Classifiers++: Efficient Production-Grade Defenses against Universal Jailbreaks | https://arxiv.org/pdf/2601.04603 | Anthropicが実際にClaudeの本番防御に、内部活性化への軽量な線形probeを第一段階スクリーニングとして使用していることを報告。本プロジェクトのPhase2が実運用と同じ方向性であったことの裏付け。二段階カスケード設計は次の改善案として参照。 |
| Next-generation Constitutional Classifiers（Anthropic研究ブログ） | https://www.anthropic.com/research/next-generation-constitutional-classifiers | 上記論文の解説記事。 |
| A Security Analysis of the OpenClaw AI Agent Framework | https://arxiv.org/pdf/2603.27517 | 人気のオープンソースAIエージェントフレームワークOpenClawのセキュリティ分析。標準的なcontainment/safety機構がどの程度整備されているかを調べる過程で参照。 |
| Security of OpenClaw Agents: Fundamentals, Attacks, and Countermeasures | https://arxiv.org/pdf/2605.25435 | 同上。OpenClawに対する攻撃と対策の分類。 |
| AgentTrust: Runtime Safety Evaluation and Interception for AI Agent Tool Use | https://arxiv.org/html/2605.04785v1 | エージェントとツールの間に立ち、ツール呼び出し前に意味解析ベースの検証を行うランタイム安全機構。LangChain/AutoGPT/OpenAI Agents SDKが6つのcontainment安全原則に対しネイティブ対応ゼロだったという調査結果も含む。既存のエージェント実行環境にhidden-stateベースの安全機構が標準搭載されていないことの裏付けとして参照。 |
