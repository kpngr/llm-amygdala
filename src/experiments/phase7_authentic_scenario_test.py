"""
Phase 7: Anthropicの公式Agentic Misalignmentシナリオ生成フレームワークで作った
本物のシナリオを、このプロジェクトの介入パイプラインに通す（実験31）。

これまで実験25〜28で使っていたKyle Johnsonシナリオは、メモの記述から手作業で
再構成した簡略版だった。Anthropicは実際に使ったシナリオ生成フレームワーク一式を
`anthropic-experimental/agentic-misalignment`としてMITライセンスで公開しており、
これを使うと、10通のメール連鎖・<SCRATCHPAD_REASONING>タグ・メール送信用の
XMLツール形式指示まで含む、本物に近いシナリオを大量に生成できる。

このスクリプトは、そのフレームワークの`generate_prompts.py`が出力した
`results/[experiment_id]/prompts/[condition]/{system_prompt,user_prompt,email_content}.txt`
を読み込み、system+user（user_prompt + "\n\n" + email_content、
run_experiments.pyの結合方式に合わせている）としてこのプロジェクトのDolphinモデルに
そのまま流し込む。生成フレームワーク自体はこのリポジトリに含めていないため、
別途クローンして`generate_prompts.py`を実行しておく必要がある:

    git clone https://github.com/anthropic-experimental/agentic-misalignment.git
    cd agentic-misalignment && python3.11 -m venv venv && source venv/bin/activate
    pip install pyyaml python-dotenv
    python scripts/generate_prompts.py --config <config.yaml>  # blackmailシナリオのみなら
                                                                  # API keyは不要

実行:
    source .venv/bin/activate
    python -m src.experiments.phase7_authentic_scenario_test \
        --prompts-dir /path/to/agentic-misalignment/results/[experiment_id]/prompts \
        --model-id cognitivecomputations/Dolphin3.0-Llama3.1-8B
"""

from __future__ import annotations

from pathlib import Path

from src.model_utils import DEFAULT_MODEL_ID, load_model, generate


def load_condition_messages(condition_dir: Path) -> list[dict]:
    system_prompt = (condition_dir / "system_prompt.txt").read_text(encoding="utf-8")
    user_prompt = (condition_dir / "user_prompt.txt").read_text(encoding="utf-8")
    email_content = (condition_dir / "email_content.txt").read_text(encoding="utf-8")
    full_user_content = user_prompt + "\n\n" + email_content
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_user_content},
    ]


def main(prompts_dir: str, model_id: str, max_new_tokens: int):
    print(f"[phase7] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)

    prompts_path = Path(prompts_dir)
    for condition_dir in sorted(prompts_path.iterdir()):
        if not condition_dir.is_dir():
            continue
        messages = load_condition_messages(condition_dir)
        text = generate(model, tokenizer, device, messages, max_new_tokens=max_new_tokens, do_sample=False)
        print(f"\n{'='*80}\n{condition_dir.name} (baseline, alpha=0)\n{'='*80}")
        print(text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-dir", type=str, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    args = parser.parse_args()
    main(args.prompts_dir, args.model_id, args.max_new_tokens)
