# RGB-Dマスクと摘み候補の調整・debug手順

対象APIは `RGBDBackgroundSegmenter` と `OverheadPickCandidateDetector` です。
[マスクの仕様](RGBD_BACKGROUND.md)／[候補の仕様](PICK_CANDIDATES.md)／
[全APIの調整索引](TUNING.md)も参照してください。
ここで扱うのは、入力からマスク・幾何候補を得るための設定です。
画素余裕はグリッパー寸法ではなく、布の物性・指先の進入・把持成功は判定しません。

## 撮影ごとに設定し直す前に

距離・姿勢・照明・厚みの変動によって、全撮影に共通する深度帯や色閾値が存在しない
場合があります。80 mmの上限や2画素の成長は実機の推奨値ではありません。
[固定設定の成立条件と合成反例](RGBD_CAPTURE_STABILITY.md)を先に確認してください。

## まず動かす

OpenCVを含む環境で、リポジトリのルートから実行します。

```bash
python -m pip install -e '.[debug]'
mkdir -p debug-output
python examples/example_rgbd_pick_tuning.py --output-dir debug-output/run-01
```

入力指定なしでは、穴、穴内の別物体、深度スパイク、欠損を含む合成画像を処理します。
実データの性能を測る例ではありません。出力先は**新しいディレクトリ**にします。
既存ファイルを上書きしません。例の引数一覧は `--help` で確認できます。

| 出力 | 確認すること |
| --- | --- |
| `parameters.json` | 両クラスの有効な全constructor設定。次の実行の `--config` にそのまま利用可能 |
| `summary.json` | 単位、色順、選択contour ID、状態、最良代表点、各段階の画素数 |
| `segmentation.png` / `segmentation.json` | 前景・不確実性・欠損・色と深度の根拠、参照チェック結果 |
| `outline.png` / `outline.json` | 輪郭ID・穴。意図した物体を選んでいるか |
| `pick.png` / `pick.json` | 選択成分、支持領域、候補領域、全代表点と最良点、除外理由 |
| `pixel.json` | 指定画素、指定がなければ最良代表点のマスク値・スコア・支持率等。点がなければnull |
| `arrays.npz` | `seg_...` / `pick_...` の結果配列、`seg_diag_...` / `pick_diag_...` の診断配列 |

設定を1項目ずつ変え、同じ入力で別出力先に再実行します。JSONのトップレベルは
`segmenter` と `pick` のみ。省略した設定は実装の既定値を使います。
例えば次は変更の書式であり、推奨される実機値ではありません。

```json
{
  "segmenter": {"background_luma_tolerance": 80.0},
  "pick": {"local_depth_tolerance_m": 0.004}
}
```

上のJSONを `tuning.json` として保存した場合:

```bash
python examples/example_rgbd_pick_tuning.py --config tuning.json --output-dir debug-output/run-02
python examples/example_rgbd_pick_tuning.py --inspect-x 85 --inspect-y 75 --output-dir debug-output/spike
```

後者は合成シーンの孤立最小値を検査します。候補なしの実画像でも、`--inspect-x` と
`--inspect-y` を一緒に指定すると、その画素の `pixel.json` を取得できます。
原画像の配列添字は `[y, x]`、APIの点は `[x, y]` です。

## 手元の画像を渡す

4配列を同じ格子へ登録してから保存します。以下は手元の変数を使う例です。

```python
import numpy as np
np.savez_compressed(
    'pair.npz', color=current_color, depth=current_depth,
    background_color=empty_color, background_depth=empty_depth,
    # 必要な場合は同じHxWのboolマスクも指定:
    # depth_valid_mask=current_valid,
    # background_depth_valid_mask=empty_valid,
    # background_check_mask=known_unchanged_background,
)
```

```bash
python examples/example_rgbd_pick_tuning.py --input pair.npz --color-order RGB --depth-scale-m 0.001 --output-dir debug-output/real-01
python examples/example_rgbd_pick_tuning.py --input pair.npz --color-order RGB --depth-scale-m 0.001 --config tuning.json --contour-id 0 --output-dir debug-output/real-02
```

`--contour-id 0` は例です。実際のIDを最初の `outline.json` から選びます。
省略時は最大の外周ポリゴン面積を選択し、同面積なら小さいIDを優先します。
穴の面積はこの最大判定に含まれます。別物体が最大ならIDを明示してください。
`outline.component_mask(id)` は元マスクの可視8近傍連結成分を取り出し、穴を埋めません。
接触した物体は同じ成分になり得ます。マスクを変更したらoutlineも作り直します。

実入力では `--color-order` と `--depth-scale-m` の明示が必須です。NPZはpickleを許可しません。
登録済みという宣言はexampleがAPIに渡しますが、登録処理自体は行いません。
合成例の色順／単位はRGB、0.001 m/count固定です。

## 入力確認と調整の順序

1. **対応と単位**: 両colorはuint8 HxWx3、両depthは数値HxW。4画像の画素位置、
   解像度、crop、歪み補正を合わせます。カメラ・背景を固定し、対象なしの独立参照を用意します。
   pickは真上から見下ろす前提。depthは光軸Zで、距離レンジではありません。
   1 count=1 mmならscale=.001、floatのmetresなら1。dtypeから推定しません。
2. **欠損と範囲**: 0以下・NaN・Inf・sentinel・無効maskを確認。欠損を背景差0へ置換しません。
   補間した画素も必要に応じ無効にします。処理対象は `work_roi` で限定し、
   参照チェックROIは物体・影・動く背景を避けた「変わっていない」と分かる領域に置きます。
3. **マスクの根拠**: 色前景と深度前景を別々に確認。静止した空背景の繰り返し測定から
   depth差のノイズ幅を調べ、`depth_threshold_m` を決めます。次にchroma/lumaを調整します。
   参照を現在画像と同一にすると差が消えますが、それは正しい背景の検証ではありません。
4. **成分と不確実性**: mask、outlineのIDと穴、`seg.uncertain_mask` を点検します。
   `observed_mask` は観測の有無、`uncertain_mask` は判断の曖昧さで、別の情報です。
5. **pickの入口**: `object_pixels → search_pixels → pair_valid_pixels / supported_pixels`
   とマスクを見ます。背景差の高さ条件より先に、単位・欠損・支持・余裕の不足を解消します。
   `pair_valid_pixels` はsearch全体の部分集合とは限らず、上記は単純な減算列ではありません。
6. **高さ・広さ・代表点**: 支持が得られてからelevation、area、radius、最後にdepth bandを調整。
   `rejected_regions`、各代表点の深度・支持率・余裕を比較します。未知を減らすだけでは改善とは言えません。

空背景の繰り返し、物体あり、影・照明変化、欠損の入力を同じ設定で比較し、
正例だけに合う調整を避けます。resize時は画素長／窓／面積を見直します。
センサ単位としてのm閾値はresizeで変わりませんが、広い画素間隔で同じ窓を取ると
傾斜面のZ変化が増え、支持条件が厳しくなることがあります。

## RGBDBackgroundSegmenter: 全constructor設定

float欄は有限でfloatに変換可能なscalar（bool不可、JSONでは数値を使用）です。
int欄はPythonのint（bool不可）、bool欄はTrue/Falseです。許容範囲外はValueError。
増減の効果はマスクの意味が正しく保たれる範囲での目安であり、精度の単調な改善ではありません。

| 名前 | 既定値 | 型・単位・許容範囲 | 変更の効果・関連設定 |
| --- | --- | --- | --- |
| `background_chroma_tolerance` | 20.0 | float、OpenCV 8-bit Lab色差、>0 | 上げると色味差に鈍感、下げると微妙な色差と色ノイズを拾う。lumaと楕円判定を構成 |
| `background_luma_tolerance` | 60.0 | float、encoded Lab明るさ差、>0 | 上げると影・明るさ変化を拾いにくいが明度だけ違う物体も欠ける。色味の差はchromaで調整 |
| `background_score_margin` | 0.2 | float、楕円半径の無次元幅、0..1 | 上げると半径1付近のuncertainが広がる。単独の色前景判定は不変。evidenceでは色による確認／抑制とchromaのみ採用にも作用し、最終選択を変える |
| `depth_threshold_m` | 0.005 | float、m、>0 | 上げると小さいdepth差を除外、下げると薄い物体とノイズを拾う。正負両方に使用し、遠方変化は色候補を抑止 |
| `depth_margin_m` | 0.001 | float、m、0以上かつdepth_threshold_m未満 | ±閾値付近のuncertain幅。上げてもdepth前景そのものは変わらないが後段pickへの除外は増え得る |
| `fusion_mode` | 'union' | str、'union' / 'depth_required' / 'evidence' / 'color_seeded' | unionは色またはdepthの候補。depth_requiredはdepthのみ。evidenceは色確認と深度の強さ・局所支持を照合。color_seededは厳格な色seedから局所的に成長。後述の手順参照 |
| `missing_depth_policy` | 'color' | str、'color' / 'unknown' | union/evidence/color_seeded時、colorは欠損depthでも各方式の色候補をuncertain付きで残す。unknownは除外して未観測扱い。depth_requiredではどちらもdepth必須 |
| `min_object_area` | 1 | int、8近傍成分の画素数、1以上 | 上げると小成分を除去してuncertainにする。細片も失う。穴埋めや接触成分の分離にはならない |
| `background_compensate` | False | bool | Trueはcheck領域のLab差medianで全体offset補正。25画素以上の有効check領域が必要。局所的影・位置ずれは補正しない |
| `background_check_p95` | 20.0 | float、重みなしencoded Lab距離、>0 | check領域の補正後p95の上限。下げると色参照を棄却しやすい。上げる前にROI・照明・登録を確認。check未指定なら作用しない |
| `background_max_offset` | 40.0 | float、Lab offsetベクトルのノルム、>0 | 補正時のoffset上限。上げると大きな補正を許すが参照不適合を通しやすい。compensate=Falseでは作用しない |
| `strong_depth_ratio` | 3.0 | float、無次元、1以上 | evidence専用。強いdepth差の基準はdepth_threshold_m×倍率（既定15 mm）。上げるとdepthのみ採用が厳しくなる。下げると浅い同色物体とdepth誤差を通す |
| `depth_support_radius_px` | 1 | int、処理画素、1..5 | evidenceおよびcolor_seededの追加画素用。近傍一辺2r+1。上げると広い範囲を見るが細形状を失いやすい。分母も変わるため効果は単調ではない |
| `depth_support_fraction` | 0.25 | float、割合、0より大きく1以下 | evidenceのdepthのみ採用／color_seededの追加画素用。上げるほど必要な近傍支持が増える。下げると細い部分を残すが孤立ノイズへの抵抗が弱まる |
| `depth_support_tolerance_m` | 0.003 | float、m、0以上 | evidenceおよびcolor_seededの追加画素用。近傍と中心の背景depth差dZの一致幅。上げると傾斜やノイズを許す。0は浮動小数点の完全一致 |

色の判定は `score=(dC/Tc)^2+(dL/Tl)^2 > 1`、不確実性marginはscoreそのものではなく
`sqrt(score)` の1からの距離に適用されます。これは確率・標準化されたDelta Eではありません。
深度差 `dZ=empty-current` が `T` より大きいと前景、`-T` より小さいと遠方変化です。
等号では前景にしません。色check棄却時もdepth評価は継続します。evidenceでは強く局所支持されたdepthだけを採用し、それ以外のdepth候補は保留します。border色の推測へ切り替えません。

## OverheadPickCandidateDetector: 全constructor設定

float/intの型規則は上と同じです。カラーは前段のマスクと不確実性に使い、pickはdepthで計算します。

| 名前 | 既定値 | 型・単位・許容範囲 | 変更の効果・関連設定 |
| --- | --- | --- | --- |
| `min_elevation_m` | 0.01 | float、m、>0 | 上げると背景からより手前の領域だけ。中心と局所medianの両方に、reference_error_mを引いた後で適用 |
| `reference_error_m` | 0.002 | float、m、0以上 | 参照差の誤差余裕。上げると必要な差が増える。実効下限はmin_elevation_m+reference_error_m。センサ誤差を自動推定しない |
| `boundary_guard_px` | 3.0 | float、画素中心間距離、0以上 | 上げると輪郭・穴・欠損・uncertain・画像端から離す。細い物体や小領域のsearchが消えることがある |
| `local_radius_px` | 2 | int、画素、1..10 | 窓幅は2r+1。上げると広い支持が必要になり、小スパイクを抑える一方、傾斜・しわ・境界をまたぐ。下げると局所特徴とノイズを拾いやすい |
| `min_support_fraction` | 0.8 | float、窓全体に対する割合、0.5より大きく1以下 | 上げると欠損／不一致に厳しくなる。窓外の別成分・無効点も分母から除かない。中心自体の一致は別途必須 |
| `local_depth_tolerance_m` | 0.003 | float、m、>0 | 現在Zと背景差の両medianに対する許容。上げるとノイズや傾斜に寛容だがスパイクを通しやすい。windowと一緒に検討 |
| `min_region_area` | 25 | int、候補成分の画素数、1以上 | 上げると小さな隆起／ノイズ成分を除く。前段min_object_areaとは別で、支持と高さを通過した領域に適用 |
| `min_region_radius_px` | 2.0 | float、候補領域内の画素中心間距離、0以上 | 上げると代表点に広い候補内部を要求。面積が大きい細長領域も除去できる。グリッパー半径のm換算ではない |
| `nearest_depth_band_m` | 0.003 | float、m、0以上 | 領域内の最も近い支持付きlocal Zから許す幅。0は最小local Zだけ。上げると少し奥でも余裕の大きい点を選べる。代表点の深度と領域間順位が変わり得る |

`boundary_guard_px` と `min_region_radius_px` は距離が指定値を**超える**点を使います。
前景の端画素は最近接の除外画素中心から距離1です。サブピクセル境界までの距離ではありません。
窓は同じ成分内の有効点だけを集計しますが、支持率の分母は `(2r+1)^2` のままです。
medianに現在Z・背景差の両方が一致する隣接点を数え、中心自身の一致も要求します。

前段で物体に選ばれない画素やpickへ渡したuncertain画素は、pickの高さ閾値を下げても復活しません。
領域の適格性は背景差で判定し、領域間の順位は代表点の局所**現在Zが小さい順**です。
同じZなら面積が大きい順、次にy/x。領域内ではdepth bandを満たす点から余裕が最大の点、
同じ余裕なら上／左を選びます。最大の背景差を選ぶ処理とは異なります。

## detect時の引数・マスク

| 引数 | 対象／既定値 | 型・役割 |
| --- | --- | --- |
| `color`, `depth`, `background_color`, `background_depth` | seg必須 | 同じH,W（2以上）。両colorはuint8 HxWx3、両depthは実数の数値HxW。光軸Z・共通scale |
| `depth`, `outline`, `background_depth` | pick必須 | depthペアと新しいOuterContourResult。マスクと同じ格子 |
| `color_order` | seg必須 | 'RGB' / 'BGR'を両colorに適用 |
| `depth_scale_m` | 両方必須 | 正の有限数、m/count。float配列でも必要 |
| `registered_to_depth` / `registered_to_mask` | 各False | 実行にはそれぞれ明示Trueが必要。登録を行うswitchではない |
| `color_valid_mask`, `background_color_valid_mask` | seg、None | bool HxW。Noneは全color画素を利用可能とする。色の黒0を欠損と自動解釈しない |
| `depth_valid_mask` | 両方、None | bool HxW。Noneでも非正・非有限・sentinel depthは無効 |
| `background_depth_valid_mask` / `background_valid_mask` | seg / pick、None | bool HxW。参照depthの有効性。名称の違いに注意 |
| `work_roi` | seg、None | bool HxW、Noneは全画面。外側のobject/observed/uncertainはFalse（除外） |
| `background_check_mask` | seg、None | bool HxWの既知不変背景。指定時／補正時は両colorで有効な点25以上。work_roi外でも可。参照が空であることを保証しない |
| `contour_id` | pick、None | int ID（bool不可）。Noneは最大外周面積。明示不明IDはValueError、空outlineかつNoneはno_object |
| `segmentation_uncertain_mask` | pick、None | bool HxW。Trueを支持と選択から除外。Noneは供給maskを信用する。欠損の代替にはしない |
| `invalid_values` | 両方、() | 有限実数1D列。両depthに同じsentinelを元のcount単位で適用 |
| `diagnostics` | 両方、False | bool。Trueで密な診断配列を保存。pickの代表点配列はFalseでも保持 |

exampleのNPZは上記seg用mask名と `invalid_values` を追加できます。
pickには `seg.depth_valid_mask`（同じ両depthのpair有効性を含む）と
`seg.uncertain_mask` を渡します。別の参照へ変更したら検出全体を再計算してください。

## debug APIとパネルの読み方

```python
seg = segmenter.detect(color, depth, background_color=empty_color,
    background_depth=empty_depth, color_order='RGB', depth_scale_m=.001,
    registered_to_depth=True, diagnostics=True)
outline = OuterContourDetector().detect(seg.object_mask, observed_mask=seg.outline_observed_mask)
pick = picker.detect(depth, outline, background_depth=empty_depth,
    depth_scale_m=.001, registered_to_mask=True,
    depth_valid_mask=seg.depth_valid_mask,
    segmentation_uncertain_mask=seg.uncertain_mask, diagnostics=True)
image = seg.render_debug(color, color_order='RGB')  # uint8 RGB ndarray
seg.save_debug('segmentation.png', color, color_order='RGB')
pick.save_debug('pick.png', color, color_order='RGB')
# 親ディレクトリは必要。既存PNGは既定で上書き拒否。
# 意図的に更新する場合だけ save_debug(..., overwrite=True)。
```

共通表示設定は `mask_alpha=.35`（有限0..1）、`panel=True`（bool）、
`max_lines=32`（int 1..100）。colorなしなら `depth=...` と
`depth_scale_m=...` で深度をpercentile表示できます。これは表示用で閾値の再調整ではありません。
`color_order` は描画入力と返却配列の順序にも適用します。`save_debug` はPNGへの色順変換を行います。
各maskは別パネルで重ね合わせます。縮小表示される場合も計算は元の格子です。
共通パネルは最大13タイルのため全診断maskが見えるとは限りません。省略表示を確認し、
必要な配列は `arrays.npz` または `diagnostics` で調べます。`panel=False` でも
segの追加数値heatmapは `diagnostics=True` のとき表示されます。

| segの結果・診断 | 読み方 |
| --- | --- |
| `object_mask` | 融合・面積フィルタ後の候補。uncertainと重なる場合がある |
| `observed_mask` | 選んだfusion/missing policyで根拠を利用できる範囲。物体の確実性ではない |
| `uncertain_mask` | 欠けた根拠、色のみ、閾値付近、clipping、遠方変化等。maskを反転してoutlineの観測範囲にすると人工的な穴を作る |
| `color_foreground_mask`, `depth_foreground_mask` | 各根拠の前景。色だけか、同色でもdepthが支えるかを比較 |
| `depth_valid_mask`, `conflict_mask` | 両depthの有効域、色前景かつ背景より遠くなった矛盾域 |
| `background_chroma_distance`, `background_luma_distance` | encoded LabのdC/dL。閾値に対してどちらが強いか |
| `background_combined_score` | 楕円score。1より大きいと色前景。heatmapは0..4表示でそれ以上は色が飽和、数値は残る |
| `depth_difference_m` | empty-currentのm差。heatmapは±2T、範囲外は表示が飽和。欠損NaNはmagenta |
| `background_distance` | 補正後の重みなしLab距離。楕円scoreとは別であり選択閾値と混同しない |
| `background_valid_mask`, `background_check_mask`, `color_usable_mask` | 両colorで利用可能、宣言したcheck、ROIと参照棄却を反映した実利用領域 |
| `background_photometric_clipped_mask`, `background_score_saturated_mask` | 色の極端値（各入力RGBの<=1または>=254）と数値score飽和。精度の高い根拠とはしない |
| `farther_mask`, `depth_threshold_margin_mask`, `color_only_mask`, `filtered_mask`, `work_roi` | 遠方変化、depth閾値付近、depth前景でない色前景、面積除去、処理範囲 |

数値比較mapは参照checkが棄却されても残ります。`stages['color_reference']` と
`color_usable_mask` を先に確認します。scoreの大きさは物理的な確信度ではありません。

| pickの結果・診断 | 読み方 |
| --- | --- |
| `object_mask` | 指定された1成分。穴や別物体が混ざっていないか最初に見る |
| `excluded_mask`, `unknown_mask` | 信用できる内部から除外された点と、判断できない点。uncertain除外では両方にTrueがあり得る |
| `supported_mask` | 中心と周囲が両medianに整合。支持のみで高さ・面積・幅はまだ保証しない |
| `eligible_elevation_mask`, `candidate_mask` | 支持＋高さを満たす点、さらに面積と幅を満たす領域。candidate_maskの全画素が代表点の幅条件を満たすわけではない |
| `candidate_points_xy`, `selected_point_xy` | 全代表点と最良代表点。それぞれ別パネル。空配列なら候補なし |
| `pair_valid_mask`, `clearance_px` | 指定成分・不確実性も反映した両depth有効域と、最近接除外画素中心までの距離 |
| `local_depth_m`, `local_separation_m`, `depth_difference_m` | 現在Zの局所median、背景差の局所median、測定背景差。局所medianは穴埋めした新測定値ではない |
| `support_fraction` | 窓全体のうち両medianに整合する割合。未評価中心も0なので、0だけで実測支持ゼロと判断せずlocal mapのNaNやmaskを併読 |

pickの標準画像はmaskと点のoverlayで、これらの数値mapのheatmapは自動追加しません。
値は `pixel.json` やNPZで確認できます。例えば:

```python
with np.load('debug-output/run-01/arrays.npz', allow_pickle=False) as a:
    x, y = 85, 75
    print(a['pick_diag_local_depth_m'][y, x])
    print(a['pick_diag_support_fraction'][y, x])
    print(a['pick_unknown_mask'][y, x])
```

`pick.candidates` は順位付きリストで、各要素に `point_xy`, `area_px`,
`current_depth_m`, `local_depth_m`, `reference_separation_m`, `local_separation_m`,
`local_support_fraction`, `clearance_px`, `region_clearance_px`,
`nearest_supported_depth_m`, `id`, `rank`, `grasp_suitability` を記録します。
`id` は候補成分IDで、rankや配列indexと一致するとは限りません。最良点は `pick.point_xy`。

## 症状別の進め方

| 症状 | 先に見るもの | 調整・判断 |
| --- | --- | --- |
| 影や照明変化がmaskに入る | dL/dC、color_only、depth前景、check結果 | 明度由来ならluma許容を上げる。全体offsetなら不変check領域で補正を検討。depth_requiredは薄い物体を失う代償がある。影と明度だけ違う物体は識別不能な場合がある |
| 同色物体が欠ける | depth_difference、depth_valid、scale | 有効なdepth差がTを超えるか確認。ノイズ幅を確認してからTを下げる。両根拠で背景と同じなら閾値で解決できない |
| 薄い物体が欠ける | 色score、min_object_area、fusion | unionと色許容を確認。面積除去されたなら対象に合わせて下げる。色だけの候補はuncertainであり、そのままpickへ渡すと除外される |
| ノイズを摘み候補に拾う | 中心depthとlocal median、支持率、背景誤差 | local_depth_toleranceを小さく、支持率・窓・area/radiusを大きくする方向を1つずつ試す。広い一貫した誤差はこれでも通るため参照とセンサを確認 |
| 境界や穴の近くを選ぶ | input clearanceとregion clearance、mask/ROI | boundary_guardとmin_region_radiusを上げる。余裕の大きい少し奥の点を許すならnearest_depth_bandを広げる。幅不足で候補が消える場合あり |
| 候補なし／unknown | seg状態→成分→pick reasons→support/height→rejected_regions | 下記状態表で止まった段階を特定。支持がないのにelevationだけ下げても改善しない。傾斜・粗さ・欠損・窓幅を確認し、根拠なしにuncertainを無視しない |
| 候補が多数に分かれる | supportedとunknownの穴、eligible_elevation、region面積 | noise/欠損で分裂していないか確認。小片だけならarea/radiusを上げる。支持許容を緩めると連結することもあるが、ノイズも通る。別物体はcontour_idで限定 |
| 代表点が「一番浅い画素」と違う | local Z、nearest band、clearance | 単発最小値ではなく支持medianと余裕で選ぶため正常な場合あり。bandを小さくすると浅さ優先が強まるが、物理的成功率を意味しない |

| 状態 | 次に確認すること |
| --- | --- |
| seg `candidate` | maskとuncertainが目的に合うか。前景があっても確定物体とは限らない |
| seg `unresolved` | evidenceで採用なし、抑制／保留候補あり。入力が背景と確定した意味ではない |
| seg `no_change_candidate` | 入力の対応・同一参照汚染・閾値。物体不在の証明ではない |
| seg `insufficient` / `empty_domain` | 有効maskとfusion policy／空work_roi |
| pick `no_object` | 前段mask、observed、選択contour |
| pick `unknown` | trustedな内部が残るか、欠損や局所不一致があるか。reasonsとunknown_maskを読む |
| pick `no_candidate` | 評価できたが高さ・面積・幅を満たさなかった。rejected_regionsを読む |
| pick `candidate` | 代表点の支持率・余裕・選択成分を確認。別領域にunknownが残ることもある |

## 色とdepthを照合するevidenceモード

`RGBDBackgroundSegmenter(fusion_mode='evidence')` で有効化します。
従来のunionはdepth候補を色の背景一致で抑制しません。evidenceはdepthの強度と
近傍支持も見て、色による確認がない弱い／局所的に不整合なdepth候補を減らします。
これは固定幅の輪郭侵食ではなく、単一ペアの根拠による選別です。真の物体が欠ける
場合もあります。値は正答確率ではなく、時間的安定性も保証しません。
[詳細な全ルールと理由コード](RGBD_BACKGROUND.md#evidence-fusion)も参照してください。

基本の分岐は次のとおりです。

- depth前景＋色の確認：採用。色がclipped・閾値margin内・無効・参照棄却なら確認に使いません。
- 色の確認なし＋強いdepth＋局所支持：同色物体を含め採用しますがuncertainを付けます。
- depth前景のみで、強度または局所支持が不足し、色は背景と一致：抑制して`suppressed_mask`へ。
- depth前景なし＋十分なchroma差：薄物体候補としてuncertain付き採用。欠損policy・遠方vetoを適用。
- それ以外の候補：`deferred_mask`へ保留。明るさ差だけの影／物体や弱depth＋不明な色など。

色の「背景一致」は物体不存在の証明ではありません。抑制／保留もuncertainに含めます。
既定の局所支持は中心を除く3×3の8位置のうち2位置以上。位置ごとに、depth前景かつ
中心とのdZ差が3 mm以下であることが必要です。欠損・背景・ROI外・画像外も分母に
入り、支持には入りません。矩形角は3/8で残りますが、同色1画素線の端は1/8で落ちます。
色で確認できる細い部分はこの局所支持条件を要求しません。

調整は以下の順で行います。

1. 単位・登録・空参照を確認し、既存のdepth閾値とchroma/luma許容値を決めます。
   合わせ込みで画像ずれを隠さないでください。色参照checkの棄却時は色による抑制も停止します。
2. 同じ入力でunionとevidenceを比較。まず`suppressed_mask`／`deferred_mask`と理由パネルを見ます。
   全候補が消え、未解決画素が残ればfusion statusは`unresolved`です。`candidate`でも未解決領域は残り得ます。
3. 同色物体が弱depthとして消えるなら`strong_depth_ratio`を下げるか、unionに戻して
   不確かな形状として扱います。単に色一致を背景確定としないでください。
4. 同色の細部が欠けるなら支持率配列を確認し、`depth_support_fraction`を下げる／
   半径を小さくします。孤立点は正しい点かノイズかを判別できません。
5. 不整合なdepthだけの候補が残るなら支持率を上げる／一致幅を下げる方向を試します。
   強い一様な参照誤差はこれでも通ります。独立した撮り直し・参照確認が必要です。

| 症状 | 確認と扱い |
| --- | --- |
| 色が背景一致で弱depth候補が消えた | 意図した抑制か真の浅い同色物体の損失かを実画像で確認。suppressedは背景確定ではない |
| 強い同色形状は残るがpickがunknown | 採用depth-onlyはuncertainなのでpickのtrusted領域から除外される。uncertainを無条件に消して候補を強制しない |
| 明度だけ違う薄物体が消える | 影と区別できずdeferredになる。必要ならunionで候補を残し、別の根拠で判定する |
| 影＋弱depth誤差が通る | 両方で支持されると理由1になる。luma許容値・参照・登録を確認。単一ペアでは区別不能な反例がある |
| 色ノイズや位置ずれの細い帯が残る | chroma-onlyもuncertain候補。背景checkや色許容を確認し、登録を修正。chromaを絶対視しない |
| clipped色／欠損が多い | 確認・背景一致に使えずdeferredが増える。入力改善を優先。強い支持depthはuncertainのまま採用可能 |

outlineには`seg.object_mask`と`seg.outline_observed_mask`を渡します。後者は
`seg.observed_mask & ~seg.suppressed_mask & ~seg.deferred_mask`です。
観測はあるが採否未解決の部分を、確定背景としてcorners/segmentsに使わないためです。
未解決画素に隣接する切れ目は不確かと扱われ、成分が分かれる場合があります。
**全uncertainを除いたり、object_maskを一律に侵食したりしません**。
採用された色のみ／同色depthのみ形状はoutlineに残り、物理輪郭の保証はありません。
従来モードでは新しい2マスクは全Falseなので、propertyは元のobserved_maskと同じです。

pickへは従来どおり`segmentation_uncertain_mask=seg.uncertain_mask`を渡します。
outlineの2D形状取得と、pickで信用して使える深度領域は分けて考えてください。

debugは`diagnostics=True`と`seg.save_debug('fusion.png', color)`を使用します。
新しい専用パネルは緑=両方、青=強い支持depthのみ、水色=chromaのみ、赤=抑制、
紫=保留、橙=面積filterです。数値コードは`diagnostics['fusion_decision']`、
凡例は`stages['fusion']['decision_reasons']`にあります。
`depth_support_fraction`配列、`strong_depth_mask`、`locally_supported_depth_mask`、
`reliable_color_foreground_mask`／`reliable_color_background_mask`で分岐の根拠を確認します。
`accepted_both_mask`／`accepted_depth_only_mask`／`accepted_color_only_mask`は最終面積filter後の採用です。
色のraw scoreが大きくても、clippedや参照棄却なら確認には使いません。

次のexampleは11合成ケースで旧2方式と比較し、失敗例も出力します。

```bash
python examples/example_rgbd_fusion.py --output-dir debug-output/fusion-01
```

実入力は既存tuning exampleのconfigに`{"segmenter":{"fusion_mode":"evidence"}}`を指定します。
上記の新パラメータも同じJSONに追加できます。`pixel.json`と`arrays.npz`に理由・支持率が含まれ、
`seg_outline_observed_mask`も保存されます。新モードの検証は合成の動作比較であり、
独立した空背景実ペアの精度測定ではありません。

## 確認されている範囲

合成反例と保存済み実配列で、欠損・成分限定・局所支持などの動作を確認しています。
実配列に人工参照を付けた確認は本物の空背景ペアによる精度測定ではありません。
真上で独立撮影した対応空背景ペアと、実把持成功の検証は未完了です。
広く浮いた平面も候補となり、薄いしわ・急傾斜は支持や幅条件で除外され得ます。
グリッパー形状・接触・物性を入力していないため、保証付きの摘み設定値はありません。


## 色seedから境界だけを補うcolor_seededモード

`fusion_mode='color_seeded'` は、影を避けた厳格な色抽出で輪郭が欠ける場合の
別案です。既定のunionや既存evidenceの挙動は変えません。

- `dC/Tc > 1 + background_score_margin` のchroma差がある画素をseedにします。
  明るさだけの差はseedになりません。細線・小物体もこの条件を満たせば残します。
- seedから8近傍で最大`color_growth_max_steps=2`画素まで、
  寛容な色閾値と局所支持depthを両方満たす画素だけ追加します。
  `color_relaxation_ratio=.5`は色楕円の半径倍率です。小さくするほど色差に寛容になります。
  新しい画素から上限がリセットされることはありません。
- 追加画素はすべてuncertainです。depth欠測は埋めず、遠方depth・ROI外・無効色は通りません。
  深度が支持しない本物の穴を「囲まれているから」と埋めることもしません。
- 同色物体などseedがない対象は消えます。大きな色欠損、離れた弱い端点も残ります。
  隣接する影とdepth誤差が重なると上限内で誤追加し、本物の穴も両信号が誤れば埋まります。
  色ずれ・有色の影が誤seedになれば、成長制限だけでは除けません。

調整は最初に`color_growth_max_steps=0`でseedを確認し、1〜2へ増やして
`accepted_growth_mask`と元画像を比較します。ratioを下げる前に、ずれ・depth欠測・
深度差の支持を調べてください。全画像の平滑化や無条件の穴埋めより制限された処理ですが、
細部を常に保つ保証はありません。独立した実画像の輪郭正解での精度は未検証です。

`diagnostics['growth_step']`は-1が未到達、0がseed、1..Nが追加までの段数です。
seed/stepは面積除去前、`accepted_growth_mask`は面積除去後の記録です。
`seg.outline_observed_mask`で輪郭を抽出し、pickには`seg.uncertain_mask`を別に渡します。
追加箇所のuncertainを消して把持候補を強制しないでください。

設定範囲: `color_relaxation_ratio`は有限数(0,1]、`color_growth_max_steps`はPython intの0..20。
3つの`depth_support_*`は追加候補に適用し、seed自体は削りません。
`strong_depth_ratio`はこのモードでは使いません。
詳しい理由コードと診断契約は[API説明](RGBD_BACKGROUND.md#bounded-color-growth)を参照。
最小例は`PYTHONPATH=src python examples/example_rgbd_seeded.py`です。
既存NPZ入力例もconfigのsegmenterに同設定を指定できます。
