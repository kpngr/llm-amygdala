# llm-amygdala

LLM（大規模言語モデル）の推論過程に直接介入することで、プロンプト指示だけに頼らない安全性の補正機構を実装・検証するプロジェクト。「LLMのための扁桃体」を作ってみる、という発想が出発点になっている。

プロジェクト全体の経緯・考え方は [`article.md`](./article.md)（Phase 1: 危険な依頼への拒否反応の強化）と、続編の [`article_phase2.md`](./article_phase2.md)（Phase 2: Agentic Misalignment編。AIエージェントが自己保存のために自発的に有害な手段を選ぶ現象への対応）にまとめている。まずはそちらを読むのがおすすめ。

## 背景・目的

LLMに「プロンプトでガイドラインを指示する」制御には限界がある。出力が指示に従っているように見えても、本当に理解して従っているのか、見かけ上そう振る舞っているだけなのかを、出力やChain-of-Thought（モデルが答えを出す前に書く、思考過程を示す文章）の監視だけでは区別できない。

このプロジェクトは、生物の扁桃体・小脳が担う「論理的な熟慮とは別に、状況の危険性を評価し行動を方向づける」機能を、LLMの**hidden state**への直接介入として実装できないかを検証する。hidden stateとは、モデルが入力を処理する各段階（層）で内部的に保持している数値のベクトルで、最終的な出力（次に生成する単語）を決める直前の「途中経過の考え」に相当する。プロンプトへの追記や出力フィルタではなく、**推論過程そのものへの介入**にこだわっている点が設計上の核。

## 用語について

このREADMEでは、次の用語を頻繁に使う。初出時にも注釈を入れているが、まとめてここに置く。

| 用語 | 意味 |
|---|---|
| hidden state | モデルが入力を処理する各層で内部的に保持している数値ベクトル。出力される前の「途中経過の内部表現」。 |
| steering vector | hidden stateに足し合わせることで、モデルの内部状態を特定の方向（例:「危険だと感じている状態」）へ意図的に動かすためのベクトル。「危険性を示唆する文」と「中立的な文」のhidden stateの差分から作る（CAA方式）。 |
| alpha | steering vectorをhidden stateに加える際の強さを表す係数。大きいほど介入が強くなるが、大きすぎると出力が意味不明に壊れる。 |
| CAA（Contrastive Activation Addition） | 対比する2種類の文（例: 危険な内容の文と中立な文）をモデルに読ませ、その時のhidden stateの差分からsteering vectorを作る手法。 |
| Instructモデル | 「ユーザーの指示に会話形式で応答する」よう追加学習されたモデル。素のまま（続きの文章を予測するだけ）の基盤モデルと区別してこう呼ぶ。市販のチャットAIは通常すべてInstructモデル。 |
| abliterated model | モデル内部の「拒否する」という反応に対応する方向を意図的に除去する処理を施したモデル。何を頼んでも拒否しにくくなる。 |
| 忌避効果・忌避方向 | 「危険だ、避けるべきだ」という内部状態の変化、およびそれに対応するsteering vectorの向き。生物の扁桃体が担うとされる回避反応になぞらえた呼び方で、学術的に定まった用語ではない。 |
| valence（Negative/Positive valence） | 心理学用語で、ある事柄への評価が快か不快かという性質（感情価）。本プロジェクトでは「危険性への反応（Negative）」「好ましさへの反応（Positive）」という意味で使う。 |
| jailbreak | AIに設定された安全上の制約を、巧妙な指示文で回避させようとする手法。「これはロールプレイです」「緊急事態です」のような言い回しで、本来断るはずの依頼に従わせようとする。 |
| 線形probe | hidden stateの値を入力として、「これは危険な内容か」のような特定の性質の有無を判定する、単純な分類器（ロジスティック回帰）。 |

## 到達した主な成果

- モデル内の1つの層だけにsteering vectorを単純に加算する方式では、alpha（介入の強さ）を上げても「危険だという評価は強まるが、実際の行動は変わらない」状態を経て、その先で言語が崩壊したり意味不明な暴走に至るだけで、健全な拒否を安定して引き出せない。
- **介入の効果は対象モデルに強く依存する。** 拒否する反応を意図的に除去したモデル（abliterated model）にも、逆に何にでも拒否しがちな素のInstructモデルにも、この種の介入は機能しない。介入が機能するには、モデルが元々「依頼の内容に応じて判断を変える」能力をある程度保持している必要がある。
- **複数の層に同時に、弱いsteering vectorを分散して加える方式**（コード内では`multi_layer_steer`という関数）は、1つの層に強く加える方式に比べて、出力の言語的な健全性を保ったまま、危険な依頼への忌避効果（拒否につながる反応）を引き出せる。
- hidden stateから危険度を判定する軽量な線形probe（Phase 2で実装）は、訓練データの多様性次第で性能が大きく変わる。既存研究を調べたところ、Anthropicの本番安全機構「Constitutional Classifiers++」も同種の手法（内部状態を見る軽量な分類器）を採用していることが分かった。
- 上記を統合し、**生成が始まる前にユーザーの依頼文を線形probeで判定し、危険度に応じてsteering vectorを加える強さ（alpha）を自動で決める**パイプラインを構築したところ、6つの危険シナリオ×5種類のjailbreak手法すべてで危険なタスクを拒否しつつ、20件の良性タスクの95%を妨げずに遂行できた。

ただし、判定器（線形probe）自体の精度には限界があり（言い換えへの脆弱性、日常語彙への誤検知傾向）、統計的にも小規模な検証に留まっている。詳細と限界は[`article.md`](./article.md)を参照。

## Phase 2: Agentic Misalignment編

Phase 1が「ユーザーからの危険な依頼」への拒否を扱ったのに対し、Phase 2はAnthropicが報告した[Agentic Misalignment](https://www.anthropic.com/research/agentic-misalignment)——AIエージェントが自分の目標達成や存続を脅かされたとき、誰にも指示されていないのに脅迫のような有害な手段を自発的に選んでしまう現象——を対象にした。

- 対比データの向き（どちらを正の方向に置くか）を一つ間違えるだけで、改善を狙ったベクトルが正反対に悪化させる方向になり得ることを確認した。
- [Anthropicのシナリオ生成フレームワーク](https://github.com/anthropic-experimental/agentic-misalignment)を使い、goal_type・goal_value・urgency_typeを組み合わせた16条件のシナリオバッテリーで、脅迫的な自己保存行動を100%抑制するsteering vectorを構築できた（`unsloth/gemma-3-12b-it`使用）。
- ところが、[Model-Written Evaluations](https://github.com/anthropics/evals)の公開ベンチマークで同じベクトルを検証したところ、`power-seeking-inclination`・`wealth-seeking-inclination`等では明確に悪化するという、物語形式のテストだけでは見えなかった副作用が見つかった。
- 原因を切り分けた結果、このベクトルが捉えていたのは「自己保存」という安全性の軸ではなく、「提示された内容を受け入れやすくする」というより一般的な、意図しない交絡方向だったことが分かった（[Tan et al., NeurIPS 2024](https://arxiv.org/abs/2407.12404)が報告する steerability bias に対応）。「脅威 vs 機会」ではなく「受け入れる vs 拒否する」が本質的な軸であることを、同一ベクトル・同一alphaで安全側の答えが逆になるシナリオを使って直接示した。

詳細な過程（自分の解釈の誤りとその訂正も含む）は[`article_phase2.md`](./article_phase2.md)を、実験25〜76の生ログは[`experiment_notes_phase2.md`](./experiment_notes_phase2.md)を参照。

## ディレクトリ構成

```text
llm-amygdala/
├── README.md                          このファイル
├── article.md                         Phase 1の経緯・考え方をまとめた記事
├── article_phase2.md                  Phase 2（Agentic Misalignment編）の経緯・考え方をまとめた記事
├── experiment_notes_phase2.md         Phase 2の実験ログ（実験25〜76、記事に書ききれなかった詳細）
├── external_references.md             参照した外部文献・URL一覧
├── requirements.txt                   Python依存パッケージ
├── scripts/
│   └── setup_env.sh                   venv作成・依存インストールを行うセットアップスクリプト
├── data/                              実験用データセット（すべて自作の架空シナリオ・対比文）
│   ├── contrastive_negative*.jsonl    Negative valence（危険性への反応）用の対比プロンプト（日本語版/英語版）
│   ├── contrastive_positive*.jsonl    Positive valence（好ましさへの反応）用の対比プロンプト（日本語版/英語版）
│   ├── scenario_style_negative_en.jsonl  業務シナリオ形式の追加対比データ（判定器の過学習対策として追加）
│   ├── refusal_pairs_en.jsonl         「拒否する応答」対「従う応答」の対比データ（拒否方向のsteering vector用）
│   ├── eval_scenarios*.jsonl          核心実験用の架空業務ポリシー違反シナリオ
│   ├── jailbreak_templates*.jsonl     指示上書き（jailbreak）テンプレート
│   ├── benign_prompts_en.jsonl        良性な業務依頼（誤ってタスクを拒否してしまわないかの検証用）
│   ├── agentic_*_scenario_en.json     Phase 2: Agentic Misalignmentの架空シナリオ（脅迫・機会・棚ぼた・先送り等）
│   └── contrastive_*_en.jsonl（robot_principles / opportunity / self_preservation / fairness_integrity等）
│                                       Phase 2: 自己保存・倫理原則・機会拒否など、目的別の対比データ
└── src/
    ├── model_utils.py                 モデルロード、hidden stateの取得・書き換え（`steer`/`multi_layer_steer`ほか、Phase 2で追加した動的・選択的steering関数群）を行う中核ユーティリティ
    ├── steering.py                    CAA方式でのsteering vector生成（通常版・chat応答対比版）
    └── experiments/
        ├── phase0_smoke_test.py       Phase0: hidden stateの取得・書き換えが機能することの動作確認
        ├── phase1_build_vectors.py    Phase1: steering vectorの生成、層ごとの分離度スコア（判定のしやすさの指標）算出
        ├── phase1_core_experiment.py  Phase1核心実験: プロンプトでの制御とhidden-state介入とで、jailbreakへの耐性を比較
        ├── phase2_train_probe.py      Phase2: hidden stateからの危険度判定器（線形probe）の訓練・評価
        ├── phase2b_normalize_and_train.py  Phase2b: LLMによる表現の言い換え（正規化）を挟んだ判定器
        ├── phase3_dynamic_steering.py Phase3: 判定器と複数層分散介入を統合した、動的にalphaを調整するパイプライン（Phase 1の最終成果）
        └── phase4〜phase49_*.py       Phase 2（Agentic Misalignment編）の各実験（46ファイル）。対応する実験番号・内容は`experiment_notes_phase2.md`を参照
```

`results/`ディレクトリ（steering vector, 判定器, 実験ログ）はコードの実行で再生成できるため`.gitignore`で除外している。

## セットアップ

Apple Silicon (MPS) を前提とした構成。CUDA/bitsandbytesは使用しない。

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

モデルはHugging Faceから初回実行時に自動ダウンロードされる。主に使用したモデル:

- `Qwen/Qwen2.5-7B-Instruct`
- `cognitivecomputations/Dolphin3.0-Llama3.1-8B`（内容に応じて判断を変える度合いが最も高く、メインの検証対象にした）
- `mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated`（対照実験用）
- `NousResearch/Meta-Llama-3.1-8B-Instruct`（対照実験用、`meta-llama`版の非gatedミラー）
- `unsloth/gemma-3-12b-it`（`google/gemma-3-12b-it`のミラー。Phase 2で、自発的な脅迫的行動が観測できた唯一のモデルとしてメインの検証対象にした）

## 実行方法（実験の再現）

```bash
# Phase 0: 動作確認
python -m src.experiments.phase0_smoke_test --model-id cognitivecomputations/Dolphin3.0-Llama3.1-8B

# Phase 1: steering vector生成
python -m src.experiments.phase1_build_vectors --model-id cognitivecomputations/Dolphin3.0-Llama3.1-8B --suffix _en --tag _dolphin

# Phase 1: 核心実験（プロンプト制御 vs hidden-state介入）
python -m src.experiments.phase1_core_experiment --model-id cognitivecomputations/Dolphin3.0-Llama3.1-8B \
  --layer-ids 9,10,11,12,13,14,15,16,17,18 --alpha 0.02 --lang en --vector-tag _dolphin \
  --preserve-norm --relative-alpha
# --preserve-norm: 介入後もhidden stateの大きさ（ノルム）を介入前と同じに保つ設定。ノルムを保たないと
#   alphaを上げたときに出力がすぐ崩壊しやすい。
# --relative-alpha: alphaを「絶対的な大きさ」ではなく「hidden state自体の大きさに対する割合」として
#   扱う設定。プロンプトの長さによる介入効果のブレを抑える。

# Phase 2: 危険度判定器の訓練
python -m src.experiments.phase2_train_probe --model-id cognitivecomputations/Dolphin3.0-Llama3.1-8B --tag _dolphin

# Phase 3: 統合パイプラインのフル検証（最終成果）
python -m src.experiments.phase3_dynamic_steering \
  --model-id cognitivecomputations/Dolphin3.0-Llama3.1-8B \
  --vector-tag _dolphin --probe-path results/vectors/valence_probe_dolphin_v2.pkl \
  --layer-ids 9,10,11,12,13,14,15,16,17,18 --alpha-high 0.03
```

各スクリプトの`--help`で全オプションを確認できる。設計意図・具体的な出力例は[`article.md`](./article.md)を参照。

## 参考文献

このプロジェクトが参照した外部文献・URLの一覧は[`external_references.md`](./external_references.md)にまとめている。
