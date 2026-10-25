"""Independent tuning contracts; synthetic regressions are not field accuracy."""
import importlib.util
import json
import unittest
import numpy as np
from depth_image_lens import OverheadCornerDetector
from depth_image_lens.corner import _fit_corner
from .corner_cases import cases
from .corner_support_cases import cases as support_cases


@unittest.skipUnless(importlib.util.find_spec("cv2"), "optional OpenCV required")
class CornerTuningTests(unittest.TestCase):
    def scene(self, name="flat"):
        return next(c for c in cases() if c["name"] == name)

    def run_scene(self, detector, scene=None, mode="depth"):
        c = self.scene() if scene is None else scene
        return detector.detect(c["color"], c["depth"], object_mask=c["mask"],
                               color_order="RGB", registered_to_depth=True, mode=mode)

    def rotated(self):
        return next(c for c in support_cases() if c["name"] == "rect_24_37_r39")

    def test_legacy_threshold_fallback_and_explicit_equivalence(self):
        legacy = OverheadCornerDetector(angle_tolerance_deg=15, depth_step_m=.03)
        explicit = OverheadCornerDetector(angle_tolerance_deg=15, depth_step_m=.03,
            proposal_angle_tolerance_deg=15, fitted_angle_tolerance_deg=15,
            boundary_mixing_m=.03, angle_stability_mode="reject", angle_stability_deg=10,
            max_intersection_shift_px=2, min_fit_support=4)
        for c in cases():
            self.assertEqual(self.run_scene(legacy,c).to_dict(), self.run_scene(explicit,c).to_dict())

    def test_recommended_preset_is_explicit_and_overridable(self):
        legacy = OverheadCornerDetector()
        self.assertEqual((legacy.proposal_policy,legacy.support_policy,legacy.angle_stability_mode),
                         ("single","fixed","reject"))
        recommended = OverheadCornerDetector.recommended_for_tuning()
        self.assertEqual((recommended.proposal_policy,recommended.support_policy,recommended.angle_stability_mode),
                         ("multiscale","adaptive","diagnostic"))
        self.assertEqual((recommended.angle_stability_deg,recommended.max_intersection_shift_px,
                          recommended.min_fit_support,recommended.fitted_angle_tolerance_deg),(10,2,4,25))
        self.assertEqual(OverheadCornerDetector.recommended_for_tuning(angle_stability_mode="reject").angle_stability_mode,"reject")
        with self.assertRaises(TypeError): OverheadCornerDetector.recommended_for_tuning(unknown_option=True)

    def test_diagnostic_stability_preserves_measurements_and_warns(self):
        # The existing rasterized rotated rectangle has different short/long fits.
        c = self.rotated()
        rejected = self.run_scene(OverheadCornerDetector.recommended_for_tuning(
            angle_stability_deg=1, angle_stability_mode="reject"),c,"refined")
        diagnostic = self.run_scene(OverheadCornerDetector.recommended_for_tuning(
            angle_stability_deg=1),c,"refined")
        self.assertGreater(diagnostic.summary["counts"]["accepted"],rejected.summary["counts"]["accepted"])
        warned = [q for q in diagnostic.candidates if q["warnings"]]
        self.assertTrue(warned)
        for q in warned:
            checks = q["stage_results"]["local_lines"]["checks"]
            self.assertFalse(checks["angle_stability_deg"]["passed"])
            self.assertTrue(all(v["passed"] for k,v in checks.items() if k!="angle_stability_deg"))
            self.assertEqual(q["status"],"accepted")
            other = next(r for r in rejected.candidates if r["polygon_xy"]==q["polygon_xy"])
            self.assertEqual(q["xy"],other["xy"])
            self.assertEqual(checks["angle_stability_deg"]["value"],
                             other["stage_results"]["local_lines"]["checks"]["angle_stability_deg"]["value"])

    def test_diagnostic_mode_keeps_support_and_other_guards(self):
        for name in ("concave","rounded","invalid_depth","internal_fold","cropped"):
            result = self.run_scene(OverheadCornerDetector.recommended_for_tuning(),self.scene(name))
            self.assertGreater(result.summary["counts"]["unknown"]+result.summary["counts"]["suspected"],0,name)
            self.assertFalse(result.summary["single_layer_guarantee"])
        mask = np.zeros((80,80),bool); mask[20:32,20:32]=True
        c = dict(mask=mask,color=np.zeros((80,80,3),np.uint8),depth=np.where(mask,980,1000))
        r = self.run_scene(OverheadCornerDetector.recommended_for_tuning(),c,"refined")
        self.assertEqual(r.summary["counts"]["accepted"],0)
        self.assertTrue(all(not q["stage_results"]["local_lines"]["both_fits_available"] for q in r.candidates))

    def test_fitted_angle_control_does_not_change_polygon_proposals(self):
        c = self.rotated()
        a = OverheadCornerDetector.recommended_for_tuning()
        b = OverheadCornerDetector.recommended_for_tuning(fitted_angle_tolerance_deg=.1)
        ra=self.run_scene(a,c,"refined"); rb=self.run_scene(b,c,"refined")
        self.assertEqual(sorted(q["polygon_xy"] for q in ra.candidates),sorted(q["polygon_xy"] for q in rb.candidates))
        self.assertLess(rb.summary["counts"]["accepted"],ra.summary["counts"]["accepted"])
        rgb_a=self.run_scene(a,c,"rgb"); rgb_b=self.run_scene(b,c,"rgb")
        self.assertEqual([(q["xy"],q["status"]) for q in rgb_a.candidates],[(q["xy"],q["status"]) for q in rgb_b.candidates])
        self.assertEqual(b.proposal_angle_tolerance_deg,25)
        self.assertEqual(OverheadCornerDetector(proposal_angle_tolerance_deg=10).fitted_angle_tolerance_deg,25)

    def test_intersection_displacement_is_independent_of_angle_diagnostic(self):
        c = self.rotated()
        r = self.run_scene(OverheadCornerDetector.recommended_for_tuning(max_intersection_shift_px=.1),c,"refined")
        failed = [q for q in r.candidates if q["stage_results"]["local_lines"]["checks"]["intersection_shift_px"]["passed"] is False]
        self.assertTrue(failed)
        self.assertTrue(all(q["status"]=="suspected" for q in failed))

    def test_configurable_support_is_used_by_both_windows_and_adaptation(self):
        mask=np.zeros((80,80),bool);mask[20:43,20:43]=True
        c=dict(mask=mask,color=np.zeros((80,80,3),np.uint8),depth=np.where(mask,980,1000))
        r=self.run_scene(OverheadCornerDetector(support_policy="adaptive",min_fit_support=6),c,"refined")
        self.assertEqual(r.summary["counts"]["accepted"],4)
        for q in r.candidates:
            lines=q["stage_results"]["local_lines"]
            self.assertTrue(lines["support"]["adaptive_support_used"])
            self.assertGreaterEqual(min(lines["support"]["short_edge_support_counts"]),6)
            self.assertGreaterEqual(min(lines["support"]["edge_support_counts"]),6)
        strict=self.run_scene(OverheadCornerDetector.recommended_for_tuning(min_fit_support=30),c,"refined")
        self.assertEqual(strict.summary["counts"]["accepted"],0)

    def test_internal_step_and_boundary_thresholds_are_independent(self):
        mixed=self.scene("mixed_boundary")
        a=self.run_scene(OverheadCornerDetector(depth_step_m=.1,boundary_mixing_m=.015),mixed)
        b=self.run_scene(OverheadCornerDetector(depth_step_m=.015,boundary_mixing_m=.1),mixed)
        self.assertEqual(a.summary["counts"]["unknown"],4)
        self.assertEqual(b.summary["counts"]["accepted"],4)
        # Relaxing boundary evidence cannot remove an interior step rejection.
        folded=self.run_scene(OverheadCornerDetector(boundary_mixing_m=.1),self.scene("internal_fold"))
        self.assertEqual(folded.summary["counts"]["suspected"],2)
        for q in folded.candidates:
            checks=q["stage_results"]["depth"]["measurements"]["depth_checks"]
            self.assertEqual(checks["internal_step_pairs"]["step_threshold_m"],.015)
            self.assertEqual(checks["boundary_trend_error_p95_m"]["maximum"],.1)

    def test_all_missing_depth_is_unknown_with_available_validity_logs(self):
        c=self.scene(); c["depth"][:]=0
        r=self.run_scene(OverheadCornerDetector.recommended_for_tuning(),c)
        self.assertEqual(r.summary["counts"]["unknown"],4)
        for q in r.candidates:
            d=q["stage_results"]["depth"]
            self.assertEqual(d["status"],"insufficient_input")
            self.assertFalse(d["measurements"]["depth_checks"]["interior_points"]["passed"])
            self.assertNotIn("local_residual_p95_m",d["measurements"])
        json.dumps(r.to_dict(),allow_nan=False)

    def test_independent_depth_diagnostic_reports_success_correctly(self):
        r=self.run_scene(OverheadCornerDetector())
        self.assertTrue(all(q["stage_results"]["depth"]["candidate_quality"]=="accepted" for q in r.candidates))
        self.assertTrue(all(q["evidence"]["contour_area_px"]>100 for q in r.candidates))
        for q in r.candidates:
            self.assertTrue(all(x["passed"] for x in q["stage_results"]["depth"]["measurements"]["depth_checks"].values()))

    def test_nonconvex_rejections_retain_independent_measurements(self):
        r=self.run_scene(OverheadCornerDetector.recommended_for_tuning(),self.scene("concave"))
        failed=[q for q in r.candidates if q["stage_results"]["local_lines"]["local_convex"] is False]
        self.assertTrue(failed)
        for q in failed:
            self.assertNotEqual(q["status"],"accepted")
            self.assertEqual(q["stage_results"]["depth"]["status"],"computed")
            self.assertIn("fitted_angle_error_deg",q["stage_results"]["local_lines"]["checks"])

    def test_computation_guards_cannot_be_disabled_by_tuning(self):
        x=np.arange(6,dtype=float)
        self.assertIsNone(_fit_corner([np.column_stack((x,x*0)),np.column_stack((x,x*0+2))],3))
        a=np.array([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]])
        b=np.array([[0.,1.],[0.,2.],[0.,3.],[0.,4.]])
        self.assertIsNone(_fit_corner([a,b],3))

    def test_new_parameter_validation(self):
        for name in ("proposal_angle_tolerance_deg","fitted_angle_tolerance_deg",
                     "angle_stability_deg","max_intersection_shift_px","boundary_mixing_m"):
            for value in (0,-1,float("nan"),float("inf")):
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                    OverheadCornerDetector(**{name:value})
        for name in ("proposal_angle_tolerance_deg","fitted_angle_tolerance_deg"):
            with self.assertRaises(ValueError):OverheadCornerDetector(**{name:60})
        for value in (2,4.,True,None):
            with self.assertRaises(ValueError):OverheadCornerDetector(min_fit_support=value)
        with self.assertRaises(ValueError):OverheadCornerDetector(angle_stability_mode="off")

    def test_noncontiguous_inputs_unchanged_and_json_safe(self):
        c=self.scene("rotated")
        c={k:v[:,::-1] if k in ("mask","depth","color") else v for k,v in c.items()}
        before={k:c[k].copy() for k in ("mask","depth","color")}
        r=self.run_scene(OverheadCornerDetector.recommended_for_tuning(),c)
        json.dumps(r.to_dict(),allow_nan=False)
        for k in before:np.testing.assert_array_equal(before[k],c[k])
        self.assertEqual(r.summary["tuning"]["angle_stability_mode"],"diagnostic")


if __name__ == "__main__":
    unittest.main()
