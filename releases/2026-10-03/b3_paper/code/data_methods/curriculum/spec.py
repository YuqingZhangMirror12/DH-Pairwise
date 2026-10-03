"""Original common seam>=20%; main-damage-free final seam>=30%; light1–4px."""
SPECS = {
 "v17.5": dict(version="v17.5", weak=[4.,8.], major=[7.,15.], notch=[5.,15.],
             notch_counts=[1,2,3,4], light=[1.,4.], primary_cap=15., gap_cap=35.,
             main_coverage=[.35,.50], coverage_draw=[.40,.46]),
 "v18": dict(version="v18", weak=[5.,8.], major=[10.,15.], notch=[5.,15.],
             notch_counts=[1,2,3,4], light=[1.,4.], primary_cap=15., gap_cap=35.,
             main_coverage=[.40,.60], coverage_draw=[.46,.56]),
}
for value in SPECS.values():
 value.update(trim_range=[.25,.40], trim_raster_tolerance=.01, area_cap=.20,
              light_coverage=.70, light_tolerance=.02, pristine_min_fraction=0.,
              pristine_protection_enabled=False,
              pristine_neighborhood_radius_px=1, light_protection_guard_px=0.,
              pristine_denominator='original pre-crop common GT-supported arc, both sides unchanged',
                 light_denominator='entire current post-primary outer contour, including seam, eroded edges and new cut edges; no contact protection',
                 light_scope='whole_postprimary_contour',
                 crop_min_smaller_perimeter_fraction=.20,
                 crop_floor_stage='after all structural cuts, before primary corrosion and final light',
                 originally_short_policy='exclude from both curriculum versions, not an uncropped exception',
                 original_min_smaller_perimeter_fraction=.20,
                 final_min_original_connectable_fraction=.30,
                 final_connectable_policy='exclude structural/primary damage, allow only final1–4px light per side; original arc denominator',
              min_inherited_correspondences=4, partial_floor=.15,
              full_generation_authorized=True, review_examples_per_type=10,
              shared_reference="v14 pre-additional-cut original pair; never sequential v17/v18 cutting",
              gap_floor=5., main_damage_one_side=True)
