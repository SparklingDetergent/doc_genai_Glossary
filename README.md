# doc_genai_Glossary
用語集

以下の表は、日本語表記と英語表記の両方を示したものです。一部のキーワードは複数の英語表現が考えられるため、代表的なものを選んで記載しています。より正確な表現が必要な場合は、文脈に合わせて適切なものを選択してください。  また、括弧内の英語表記は、いくつかの候補を表している場合もあります。


## 生成AI関連キーワードリスト（技術・原理編）

| 分類 | No. | 日本語表記 | 英語表記 | 説明 |
|---|---|---|---|---|
| **基礎アルゴリズム・アーキテクチャ** | 1 | Diffusion Models | Diffusion Models | 生成AIの主要アルゴリズムの1つ。画像生成や音声合成の進化に寄与。 |
|  | 2 | Transformer Architecture | Transformer Architecture | 自然言語処理や画像生成の中心的技術。 |
|  | 3 | Attention Mechanism | Attention Mechanism | トランスフォーマーやGANで使用される、情報の重要度を計算する仕組み。 |
|  | 4 | GANs (Generative Adversarial Networks) | GANs (Generative Adversarial Networks) | 生成AIの基盤技術の一つ。生成器と識別器の競争による学習。 |
|  | 5 | ディープラーニング | Deep Learning | 機械学習の一種で、多層のニューラルネットワークを用いる手法 |
|  | 6 | トランスフォーマー | Transformer | 自然言語処理において広く用いられるアーキテクチャ |
|  | 7 | 生成的敵対ネットワーク (GAN) | Generative Adversarial Network (GAN) | 2つのネットワークを競わせることでデータを生成するモデル |
|  | 8 | 自己注意機構 | Self-Attention Mechanism | トランスフォーマーの中核となる機構 |
|  | 9 | 深層学習 | Deep Learning | 機械学習の一分野で、多層のニューラルネットワークを使用してデータを処理する手法 |
|  | 10 | ニューラルネットワーク | Neural Network | 人間の脳を模倣した情報処理モデル |
|  | 11 | 再帰型ニューラルネットワーク (RNN) | Recurrent Neural Network (RNN) | 時系列データを処理するためのニューラルネットワーク |
|  | 12 | 生成的敵対ネットワーク (GAN) | Generative Adversarial Network (GAN) | 2つのネットワークを競わせることでデータを生成するモデル |
|  | 13 | Transformerアーキテクチャ | Transformer Architecture | 自己注意機構、並列処理能力、長文処理能力に関する研究 |
| **学習手法・技術** | 14 | Fine-Tuning | Fine-tuning | 生成モデルを特定のタスクやデータセットに適応させる技術。 |
|  | 15 | LoRA (Low-Rank Adaptation) | LoRA (Low-Rank Adaptation) | パラメータ効率の良いファインチューニング手法。 |
|  | 16 | 人間からのフィードバックによる強化学習 (RLHF) | Reinforcement Learning from Human Feedback (RLHF) | ユーザーのフィードバックを用いて生成AIを強化学習する手法。 |
|  | 17 | 自己教師あり学習 | Self-Supervised Learning | ラベルなしデータから特徴を学習する手法。 |
|  | 18 | データ拡張 | Data Augmentation | 学習データを増やすための手法 |
|  | 19 | アンサンブル学習 | Ensemble Learning | 複数のモデルを組み合わせることで精度を向上させる手法 |
|  | 20 | トランスファーラーニング | Transfer Learning | 事前学習済みモデルを別のタスクに適用する手法 |
|  | 21 | 自己教師あり学習 | Self-Supervised Learning | ラベルなしデータを使用してモデルを訓練する手法 |
|  | 22 | 生成モデル | Generative Model | データの分布を学習し、新しいデータを生成するモデルの総称 |
|  | 23 | 機械学習 | Machine Learning | データからパターンを学習し、予測や意思決定を行う技術 |
|  | 24 | 深層学習 | Deep Learning | 機械学習の一種で、多層のニューラルネットワークを用いる手法 |
|  | 25 | 強化学習 | Reinforcement Learning | 環境との相互作用を通じて学習する手法 |
|  | 26 | 自己学習型AI (AutoML) | AutoML (Automated Machine Learning) | 自動で機械学習モデルを構築する技術 |
|  | 27 | 事前学習 (Pre-training) | Pre-training | 自己教師あり学習、教師あり学習、強化学習の役割と効果に関する研究 |
|  | 28 | ファインチューニング | Fine-tuning | 効率的な方法、過学習の防止、ドメイン適応に関する研究 |
|  | 29 | 強化学習からのフィードバック (RLHF) | Reinforcement Learning from Human Feedback (RLHF) | 人間によるフィードバックに基づくモデルの改善、安全性向上に関する研究 |
| **モデル評価・指標** | 30 | エントロピー | Entropy | モデルの生成結果の多様性や確率分布の不確実性を測る尺度。 |
|  | 31 | パープレキシティ | Perplexity | 言語モデルの予測精度を評価する指標。 |
|  | 32 | ハイパーパラメータチューニング | Hyperparameter Tuning | モデルのパフォーマンスを最適化するパラメータ調整 |
| **プロンプト・入力制御** | 33 | プロンプトエンジニアリング | Prompt Engineering | 生成AIモデルに対して高品質な応答を得るための入力テキストの工夫。 |
|  | 34 | プロンプトエンジニアリング | Prompt Engineering | LLMに最適な指示を与えるための手法論と設計原理 |
|  | 35 | 少ショットプロンプティング | Few-shot Prompting | 少数の例示を使ってAIにタスクを教えるプロンプト手法。 |
|  | 36 | プロンプトエンジニアリング | Prompt Engineering | 効果的なプロンプト設計、few-shot learning、zero-shot learningに関する研究 |
| **応用技術・モデル能力** | 37 | 潜在空間 | Latent Space | データの本質的な特徴が表現される空間。 |
|  | 38 | トークナイゼーション | Tokenization | テキストデータをモデルで処理可能な形に分割する技術。 |
|  | 39 | ゼロショット学習 | Zero-shot Learning | 学習していないタスクにモデルが対応できる能力。 |
|  | 40 | マルチモーダルAI | Multimodal AI | テキスト、画像、音声など複数のデータ形式を統合的に扱う技術。 |
|  | 41 | ダイナミックサンプリング | Dynamic Sampling | 生成中に出力の多様性を調整する技術。 |
|  | 42 | セマンティックコンシステンシー | Semantic Consistency | 長文生成や複雑な画像生成における意味の一貫性を保つ技術。 |
|  | 43 | 大規模言語モデル（LLM） | Large Language Model (LLM) | テキスト生成、翻訳、質問応答など、自然言語処理タスクを幅広くこなす巨大なニューラルネットワークモデル |
|  | 44 | 埋め込み | Embedding | 単語や文章などのデータを、意味的な類似性や関係性を反映した多次元のベクトル空間に変換すること |
|  | 45 | 生成AI | Generative AI | 新しいコンテンツを生成するために設計された人工知能の一種 |
|  | 46 | コンテンツ生成 | Content Generation | テキスト、画像、音声、動画などの新しいコンテンツを自動的に作成するプロセス |
|  | 47 | インコンテキストラーニング | In-context Learning | 文脈からの学習能力。追加学習なしで新しいタスクに適応する仕組み |
|  | 48 | チェインオブスロット | Chain of Thought | 段階的な思考プロセスを通じて複雑な問題を解決する推論手法 |
|  | 49 | 検索拡張型生成 (RAG) | Retrieval-Augmented Generation (RAG) | 外部知識を参照しながら生成を行う手法 |
|  | 50 | スパースアクティベーション | Sparse Activation | ニューラルネットワークの一部のみを選択的に活性化させる効率的な処理 |
|  | 51 | パラメータ効率の良いファインチューニング | Parameter-Efficient Fine-Tuning | 少ないパラメータ調整で特定タスクに適応させる手法 |
|  | 52 | マルチモーダル学習 | Multimodal Learning | テキスト、画像、音声など複数のモダリティを統合的に処理する学習 |
|  | 53 | 知識蒸留 | Knowledge Distillation | 大規模モデルの知識を小規模モデルに効率的に転移する技術 |
|  | 54 | トークナイゼーション | Tokenization | テキストをAIが処理しやすい形に分解する技術 |
|  | 55 | 自然言語処理 (NLP) | Natural Language Processing (NLP) | 人間が使う言語を理解し、生成するための技術 |
|  | 56 | 大規模言語モデル (LLM) | Large Language Model (LLM) | 大量のテキストデータを使って訓練された、強力な言語理解と生成能力を持つモデル |
|  | 57 | 少ショットプロンプティング | Few-shot Prompting | 少数の例示を使ってAIにタスクを教えるプロンプト手法 |
|  | 58 | ツリーオブソーツ (ToT) | Tree of Thoughts (ToT) | AIが問題解決やクリエイティブな思考を助けるために用いる思考の構造化方法 |
|  | 59 | 検索拡張型生成 (RAG) | Retrieval-Augmented Generation (RAG) | 生成された内容の精度を向上させるために、外部データベースから情報を取得し利用する手法 |
|  | 60 | セマンティック検索 | Semantic Search | 文脈や意味を理解した上での検索を可能にする技術 |
|  | 61 | リアルタイム高解像度画像生成 | Real-time High-Resolution Image Generation | 高解像度の画像をリアルタイムで生成する技術 |
|  | 62 | テキストプロンプトからの動画生成 | Video Generation from Text Prompts | テキストの説明から動画を生成する技術 |
|  | 63 | 自然言語処理の高度化 | Advancement in Natural Language Processing | より高度な自然言語処理技術の開発 |
|  | 64 | クロスモーダル検索技術 | Cross-modal Retrieval | 異なるデータ形式を統合的に検索する技術 |
|  | 65 | AIによる創造的ライティング | AI-powered Creative Writing | AIを用いた創作活動 |
|  | 66 | 大規模言語モデル (LLM) | Large Language Model (LLM) | そのアーキテクチャ、学習データのバイアス、スケーラビリティ、限界に関する研究 |
|  | 67 | 知識蒸留 | Knowledge Distillation | 大規模モデルの知識を小型モデルに効率的に転移する手法に関する研究 |
|  | 68 | 埋め込み表現 | Embedding Representation | 単語、文章、画像などのベクトル表現、その性質と応用に関する研究 |
|  | 69 | 生成モデルの多様性と制御 | Diversity and Control of Generative Models | 多様な出力を生成する手法、特定のスタイルや内容を制御する手法に関する研究 |
|  | 70 | 潜在空間操作 | Latent Space Manipulation | 生成されたコンテンツの編集、合成に関する研究 |
|  | 71 | マルチモーダル学習 | Multimodal Learning | テキスト、画像、音声などの異なるデータ形式を統合的に扱う手法に関する研究 |
|  | 72 | 対話型AIにおける文脈理解 | Context Understanding in Conversational AI | 長期記憶、文脈スイッチングに関する研究 |
| **安全性・倫理** | 73 | エシカルAI | Ethical AI | 生成AIの開発・利用における倫理的な課題。バイアスやフェイクコンテンツの問題を含む。 |
|  | 74 | プロンプトインジェクション | Prompt Injection | モデルの動作を予期せぬ形で操作する方法。悪意のあるプロンプトによる攻撃手法。 |
|  | 75 | バイアスと公平性 | Bias and Fairness | AIモデルにおける偏りや公平性の問題 |
|  | 76 | バイアス | Bias | AIモデルが学習データに基づいて持つ偏り |
|  | 77 | 倫理的AI | Ethical AI | AI技術の開発と利用において倫理的な考慮を重視するアプローチ |
|  | 78 | コンスティテューショナルAI | Constitutional AI | 倫理的な制約や行動指針を組み込んだAIの開発アプローチ |
|  | 79 | プロンプトインジェクション | Prompt Injection | AIの応答を操作または誤解させるために意図的に設計されたプロンプト |
|  | 80 | AIの責任性 | AI Accountability | AIシステムの行動に対する説明責任と責任の所在 |
|  | 81 | プライバシー保護 | Privacy Protection | AIシステムにおける個人情報の保護 |
|  | 82 | AIの説明可能性 | Explainable AI (XAI) | AIシステムの意思決定過程の透明性と理解可能性 |
|  | 83 | AIの公平性 | AI Fairness | AIシステムにおけるバイアスや差別を排除すること |
|  | 84 | AI倫理と公平性 | AI Ethics and Fairness | AI開発・利用における倫理的配慮と公平性の確保 |
|  | 85 | モデルの解釈可能性 | Model Interpretability | 決定過程の透明性、バイアス検出に関する研究 |
|  | 86 | バイアス検出と軽減 | Bias Detection and Mitigation | 学習データのバイアスの影響、軽減のための技術に関する研究 |
|  | 87 | フェイクコンテンツ検出 | Fake Content Detection | 生成されたコンテンツの真偽判定に関する研究 |
|  | 88 | 著作権と知的財産権 | Copyright and Intellectual Property Rights | 生成AIによる著作物の権利関係に関する研究 |
|  | 89 | プライバシー保護 | Privacy Protection | 個人情報の保護、データセキュリティに関する研究 |
| **その他** | 90 | エクサパラメータ | Exa-parameter | 10の18乗のパラメータを持つ巨大モデル |
|  | 91 | 量子コンピューティング | Quantum Computing | 量子力学を利用した計算手法 |
|  | 92 | シンセティックデータ | Synthetic Data | 人工的に生成されたデータ |
|  | 93 | 温度 | Temperature | 生成AIの出力におけるランダム性や創造性を制御するパラメータ |
|  | 94 | 人間とAIの協調 | Human-AI Collaboration | 人間とAIが効果的に協力してタスクを実行すること |
|  | 95 | 人工知能 | Artificial Intelligence (AI) | 知的な行動を模倣するコンピュータシステム |
|  | 96 | 画像認識 | Image Recognition | 画像から物体を認識・分類する技術 |
|  | 97 | ロボティクス | Robotics | ロボットの設計、制御、応用に関する技術 |
|  | 98 | クラウドAI | Cloud AI | クラウドコンピューティングを利用したAI |
|  | 99 | エッジAI | Edge AI | エッジデバイス上で動作するAI |
|  | 100 | オンデマンドAI | On-demand AI | 必要に応じてAIリソースを提供するサービス |
|  | 101 | セキュアAI | Secure AI | セキュリティを考慮したAI |
|  | 102 | エコシステムAI | Ecosystem AI | 複数のAIシステムが連携して動作する環境 |
|  | 103 | 医療診断 | Medical Diagnosis | AIを用いた医療診断 |
|  | 104 | 自動運転 | Autonomous Driving | AIを用いた自動運転技術 |
|  | 105 | ファクトチェック | Fact-checking | AIを用いた事実確認 |
|  | 106 | パーソナライゼーション | Personalization | ユーザーに合わせた個別化されたサービス |
|  | 107 | リスク管理 | Risk Management | AIを用いたリスク管理 |
|  | 108 | 量子AI | Quantum AI | 量子コンピューティングを用いたAI |
|  | 109 | 説明可能AI (XAI) | Explainable AI (XAI) | AIの意思決定過程を人間が理解できるようにする技術 |
|  | 110 | 継続的学習システム | Continual Learning System | 継続的に学習し、性能を向上させるシステム |
|  | 111 | 低リソースAI学習 | Low-resource AI Learning | データや計算資源が少ない環境でも学習可能なAI |
|  | 112 | 高度なデータ分析と予測モデリング | Advanced Data Analysis and Predictive Modeling | 複雑なデータから洞察を引き出し、予測を行う技術 |
|  | 113 | バイアス削減技術 | Bias Reduction Techniques | AIモデルのバイアスを軽減する技術 |
|  | 114 | AR/VRとAIの統合 | AR/VR and AI Integration | 拡張現実(AR)や仮想現実(VR)とAIを統合した技術 |
|  | 115 | 専門分野特化型AI | Specialized AI | 特定の専門分野に特化したAI |
|  | 116 | 対話型AIインターフェース | Conversational AI Interface | ユーザーと自然言語で対話するAIインターフェース |
|  | 117 | 因果推論 | Causal Inference | 生成AIにおける因果関係の理解と活用に関する研究 |
|  | 118 | 汎化性能 | Generalization Performance | 未知のデータに対するモデルの性能に関する研究 |
|  | 119 | ロバスト性 | Robustness | ノイズや外乱に対するモデルの頑健性に関する研究 |



## 補足

* 略語は、初めて出現した際に正式名称と併記するのが一般的です。
* カタカナ語は、文脈に応じて適切な用語に置き換えることが望ましいです。
* 表の内容は、あくまでも一般的な説明です。それぞれの用語の詳細については、専門書や論文などを参照してください。

