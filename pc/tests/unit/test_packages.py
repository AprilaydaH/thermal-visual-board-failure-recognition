"""Footprint parsing, physical scale, and size-based package matching."""

import pytest

from epr.core.domain_models.region import BoundingBox
from epr.processing.scale import (
    CalibratedScale,
    ThinLensOptics,
    box_size_mm,
    size_uncertainty_mm,
)
from epr.recognition.packages import sexpr
from epr.recognition.packages.footprints import (
    family_for_library,
    package_for_name,
    parse_footprint,
)
from epr.recognition.packages.library import PackageLibrary, SizeQuery

# An 0805 chip resistor: body 2.0 x 1.25 mm on F.Fab, courtyard 3.36 x 1.9 mm.
R0805 = """(footprint "R_0805_2012Metric"
	(layer "F.Cu")
	(descr "Resistor SMD 0805")
	(tags "resistor")
	(attr smd)
	(fp_rect (start -1.68 -0.95) (end 1.68 0.95) (layer "F.CrtYd"))
	(fp_rect (start -1 -0.625) (end 1 0.625) (layer "F.Fab"))
	(pad "1" smd roundrect (at -0.9125 0) (size 1.025 1.4) (layers "F.Cu"))
	(pad "2" smd roundrect (at 0.9125 0) (size 1.025 1.4) (layers "F.Cu"))
)
"""

# A through-hole part with no F.Fab outline: the courtyard has to serve.
COURTYARD_ONLY = """(footprint "Connector_Thing"
	(layer "F.Cu")
	(attr through_hole)
	(fp_line (start -5 -2) (end 5 -2) (layer "F.CrtYd"))
	(fp_line (start 5 -2) (end 5 2) (layer "F.CrtYd"))
	(fp_line (start 5 2) (end -5 2) (layer "F.CrtYd"))
	(fp_line (start -5 2) (end -5 -2) (layer "F.CrtYd"))
	(pad "1" thru_hole circle (at 0 0) (size 1.6 1.6) (layers "*.Cu"))
)
"""

# Neither outline, so the pads define the extent, one of them rotated.
PADS_ONLY = """(footprint "Pads_Only"
	(layer "F.Cu")
	(pad "1" smd rect (at -2 0) (size 1 3) (layers "F.Cu"))
	(pad "2" smd rect (at 2 0 90) (size 1 3) (layers "F.Cu"))
)
"""


def write(tmp_path, text, name, library="Resistor_SMD"):
    directory = tmp_path / f"{library}.pretty"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.kicad_mod"
    path.write_text(text, encoding="utf-8")
    return path


def test_sexpr_reads_nested_lists_and_quoted_strings():
    tree = sexpr.parse('(footprint "R_0805" (layer "F.Cu") (pad "1" smd))')
    node = tree[0]

    assert node[0] == "footprint"
    assert node[1] == "R_0805"
    assert sexpr.atom(node, "layer") == "F.Cu"
    assert len(list(sexpr.tagged(node, "pad"))) == 1


def test_sexpr_rejects_unbalanced_input():
    with pytest.raises(ValueError, match="unbalanced"):
        sexpr.parse("(footprint (layer))))")
    with pytest.raises(ValueError, match="unbalanced"):
        sexpr.parse("(footprint (layer)")


def test_sexpr_reads_numbers_in_order():
    tree = sexpr.parse("(fp_rect (start -1 -0.625) (end 1 0.625))")
    assert sexpr.floats(sexpr.first(tree[0], "start")) == (-1.0, -0.625)


def test_body_outline_is_preferred_over_courtyard(tmp_path):
    footprint = parse_footprint(write(tmp_path, R0805, "R_0805_2012Metric"))

    assert footprint is not None
    assert footprint.source == "fab"
    # The real 0805 body, not the 3.36 x 1.9 courtyard.
    assert (footprint.length_mm, footprint.width_mm) == (2.0, 1.25)
    assert footprint.family == "resistor"
    assert footprint.package == "0805"
    assert footprint.mounting == "smd"
    assert footprint.pad_count == 2


def test_courtyard_is_used_when_there_is_no_body(tmp_path):
    footprint = parse_footprint(write(tmp_path, COURTYARD_ONLY, "Conn", library="Connector"))

    assert footprint is not None
    assert footprint.source == "courtyard"
    assert (footprint.length_mm, footprint.width_mm) == (10.0, 4.0)
    assert footprint.family == "connector"
    assert footprint.mounting == "tht"


def test_pads_are_the_last_resort_and_rotation_is_honoured(tmp_path):
    footprint = parse_footprint(write(tmp_path, PADS_ONLY, "Pads_Only"))

    assert footprint is not None
    assert footprint.source == "pads"
    # Pad 2 is rotated 90 degrees, so it spans 3 mm in x: extent runs -2.5 to 3.5.
    assert footprint.length_mm == pytest.approx(6.0)


def test_unreadable_footprint_returns_none(tmp_path):
    assert parse_footprint(write(tmp_path, "(not_a_footprint)", "junk")) is None
    assert parse_footprint(tmp_path / "absent.kicad_mod") is None


@pytest.mark.parametrize(
    ("library", "family"),
    [
        ("Resistor_SMD", "resistor"),
        ("Capacitor_Tantalum_SMD", "capacitor"),
        ("Package_QFP", "qfp"),
        ("Package_BGA", "bga"),
        ("Package_SO", "soic"),
        ("Package_TO_SOT_SMD", "sot"),
        ("Connector_JST", "connector"),
        ("Something_Unknown", "other"),
    ],
)
def test_library_names_map_to_families(library, family):
    assert family_for_library(library) == family


@pytest.mark.parametrize(
    ("name", "package"),
    [
        ("R_0805_2012Metric", "0805"),
        ("C_0603_1608Metric", "0603"),
        ("TQFP-64_10x10mm_P0.5mm", "TQFP-64"),
        ("SOIC-8_3.9x4.9mm_P1.27mm", "SOIC-8"),
        ("SOT-23-5", "SOT-23-5"),
    ],
)
def test_package_codes_drop_dimensions_and_pitch(name, package):
    assert package_for_name(name) == package


def library():
    from epr.recognition.packages.footprints import Footprint

    def make(name, family, package, length, width, source="fab"):
        return Footprint(name, "L", family, package, length, width, source, 2, "smd")

    return PackageLibrary(
        [
            make("R_0402", "resistor", "0402", 1.0, 0.5),
            make("R_0603", "resistor", "0603", 1.6, 0.8),
            make("R_0805", "resistor", "0805", 2.0, 1.25),
            make("R_1206", "resistor", "1206", 3.2, 1.6),
            make("C_0805", "capacitor", "0805", 2.0, 1.25),
            make("SOT-23", "sot", "SOT-23", 2.9, 1.3),
            make("QFP-64", "qfp", "TQFP-64", 12.0, 12.0, source="courtyard"),
        ]
    )


def test_an_0805_matches_0805_ahead_of_its_neighbours():
    matches = library().match(SizeQuery(2.0, 1.25), families=["resistor"])

    assert matches[0].package == "0805"
    assert matches[0].score > 0.99
    assert matches[1].score < matches[0].score


def test_sizes_that_appearance_cannot_separate_are_separated_by_millimetres():
    """The case the whole module exists for: 0603 and 0805 look alike, but differ in size."""
    small = library().match(SizeQuery(1.6, 0.8), families=["resistor"])
    large = library().match(SizeQuery(2.0, 1.25), families=["resistor"])

    assert small[0].package == "0603"
    assert large[0].package == "0805"
    assert small[0].score > 0.9 and large[0].score > 0.9


def test_identical_sizes_in_different_families_both_surface():
    matches = library().match(SizeQuery(2.0, 1.25))
    top = {(m.family, m.package) for m in matches if m.score > 0.99}

    assert ("resistor", "0805") in top
    assert ("capacitor", "0805") in top


def test_orientation_does_not_matter():
    upright = library().match(SizeQuery(2.0, 1.25))
    rotated = library().match(SizeQuery(1.25, 2.0))
    assert [m.label for m in upright] == [m.label for m in rotated]


def test_a_size_nothing_matches_returns_nothing():
    assert library().match(SizeQuery(40.0, 40.0)) == []


def test_uncertainty_widens_the_field():
    confident = library().match(SizeQuery(1.8, 1.0), families=["resistor"])
    vague = library().match(SizeQuery(1.8, 1.0, uncertainty_mm=1.0), families=["resistor"])
    assert len(vague) >= len(confident)


def test_body_only_excludes_approximate_outlines():
    assert library().match(SizeQuery(12.0, 12.0)) != []
    assert library().match(SizeQuery(12.0, 12.0), body_only=True) == []


def test_a_query_must_have_a_positive_size():
    with pytest.raises(ValueError, match="positive"):
        SizeQuery(0.0, 1.0)


def test_an_empty_library_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        PackageLibrary([])


def test_library_survives_a_save_and_load(tmp_path):
    path = library().save(tmp_path / "packages.json")
    reloaded = PackageLibrary.load(path)

    assert len(reloaded) == len(library())
    assert reloaded.match(SizeQuery(2.0, 1.25))[0].package == "0805"


def test_thin_lens_scale_grows_with_distance():
    optics = ThinLensOptics(focal_length_mm=25.0, sensor_width_mm=7.4, image_width_px=2592)

    near = optics.mm_per_pixel(150.0)
    far = optics.mm_per_pixel(300.0)

    assert far > near
    # At 150 mm with a 25 mm lens the magnification is 0.2, so a 2.86 um pixel covers 14.3 um.
    assert near == pytest.approx(optics.pixel_pitch_mm * 5.0, rel=1e-6)


def test_thin_lens_rejects_a_distance_inside_the_focal_length():
    optics = ThinLensOptics(25.0, 7.4, 2592)
    with pytest.raises(ValueError, match="focal length"):
        optics.mm_per_pixel(20.0)


def test_thin_lens_validates_its_geometry():
    with pytest.raises(ValueError, match="focal length"):
        ThinLensOptics(0.0, 7.4, 2592)
    with pytest.raises(ValueError, match="image width"):
        ThinLensOptics(25.0, 7.4, 0)


def test_calibrated_scale_is_linear_in_distance():
    scale = CalibratedScale(mm_per_pixel_at_reference=0.01, reference_distance_mm=150.0)

    assert scale.mm_per_pixel(150.0) == pytest.approx(0.01)
    assert scale.mm_per_pixel(300.0) == pytest.approx(0.02)


def test_box_size_converts_to_millimetres():
    box = BoundingBox(x=0.1, y=0.1, width=0.05, height=0.02)

    length, width = box_size_mm(box, 2000, 1000, mm_per_pixel=0.02)

    assert length == pytest.approx(2.0)  # 0.05 * 2000 * 0.02
    assert width == pytest.approx(0.4)  # 0.02 * 1000 * 0.02


def test_box_size_rejects_a_nonsense_scale():
    with pytest.raises(ValueError, match="mm_per_pixel"):
        box_size_mm(BoundingBox(x=0, y=0, width=0.1, height=0.1), 100, 100, 0.0)


def test_distance_error_carries_through_proportionally():
    assert size_uncertainty_mm(20.0, 150.0, 2.0) == pytest.approx(20.0 * 2.0 / 150.0)
    assert size_uncertainty_mm(2.0, 150.0, 2.0) < 0.03
