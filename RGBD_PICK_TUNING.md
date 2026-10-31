# RGB-Dマスクと摘み候補の調整・debug手順

対象APIは `RGBDBackgroundSegmenter` と `OverheadPickCandidateDetector` です。
[マスクの仕様](RGBD_BACKGROUND.md)／[候補の仕様](PICK_CANDIDATES.md)／
[全APIの調整索引](TUNING.md)も参照してください。
ここで扱うのは、入力からマスク・幾何候補を得るための設定です。
画素余裕はグリッパー寸法ではなく、布の物性・指先の進入・把持成功は判定しません。

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
| `background_score_margin` | 0.2 | float、楕円半径の無次元幅、0..1 | 上げると半径1付近のuncertainが広がる。前景判定は変えないが、uncertainを渡すpickの候補は減り得る |
| `depth_threshold_m` | 0.005 | float、m、>0 | 上げると小さいdepth差を除外、下げると薄い物体とノイズを拾う。正負両方に使用し、遠方変化は色候補を抑止 |
| `depth_margin_m` | 0.001 | float、m、0以上かつdepth_threshold_m未満 | ±閾値付近のuncertain幅。上げてもdepth前景そのものは変わらないが後段pickへの除外は増え得る |
| `fusion_mode` | 'union' | str、'union' / 'depth_required' | unionは色またはdepthの候補。depth_requiredはdepthだけを選択し、影を抑える代わりに薄い／欠損depth物体を失う |
| `missing_depth_policy` | 'color' | str、'color' / 'unknown' | union時、colorは欠損depthでも色候補をuncertain付きで残す。unknownは除外して未観測扱い。depth_requiredではどちらもdepth必須 |
| `min_object_area` | 1 | int、8近傍成分の画素数、1以上 | 上げると小成分を除去してuncertainにする。細片も失う。穴埋めや接触成分の分離にはならない |
| `background_compensate` | False | bool | Trueはcheck領域のLab差medianで全体offset補正。25画素以上の有効check領域が必要。局所的影・位置ずれは補正しない |
| `background_check_p95` | 20.0 | float、重みなしencoded Lab距離、>0 | check領域の補正後p95の上限。下げると色参照を棄却しやすい。上げる前にROI・照明・登録を確認。check未指定なら作用しない |
| `background_max_offset` | 40.0 | float、Lab offsetベクトルのノルム、>0 | 補正時のoffset上限。上げると大きな補正を許すが参照不適合を通しやすい。compensate=Falseでは作用しない |

色の判定は `score=(dC/Tc)^2+(dL/Tl)^2 > 1`、不確実性marginはscoreそのものではなく
`sqrt(score)` の1からの距離に適用されます。これは確率・標準化されたDelta Eではありません。
深度差 `dZ=empty-current` が `T` より大きいと前景、`-T` より小さいと遠方変化です。
等号では前景にしません。色check棄却時も使えるdepthは継続し、border色の推測へ切り替えません。

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
outline = OuterContourDetector().detect(seg.object_mask, observed_mask=seg.observed_mask)
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
| seg `no_change_candidate` | 入力の対応・同一参照汚染・閾値。物体不在の証明ではない |
| seg `insufficient` / `empty_domain` | 有効maskとfusion policy／空work_roi |
| pick `no_object` | 前段mask、observed、選択contour |
| pick `unknown` | trustedな内部が残るか、欠損や局所不一致があるか。reasonsとunknown_maskを読む |
| pick `no_candidate` | 評価できたが高さ・面積・幅を満たさなかった。rejected_regionsを読む |
| pick `candidate` | 代表点の支持率・余裕・選択成分を確認。別領域にunknownが残ることもある |

## 確認されている範囲

合成反例と保存済み実配列で、欠損・成分限定・局所支持などの動作を確認しています。
実配列に人工参照を付けた確認は本物の空背景ペアによる精度測定ではありません。
真上で独立撮影した対応空背景ペアと、実把持成功の検証は未完了です。
広く浮いた平面も候補となり、薄いしわ・急傾斜は支持や幅条件で除外され得ます。
グリッパー形状・接触・物性を入力していないため、保証付きの摘み設定値はありません。
