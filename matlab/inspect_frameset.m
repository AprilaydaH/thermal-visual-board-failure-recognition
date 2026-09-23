function probe = inspect_frameset(projectDir, x, y, frameSetId)
%INSPECT_FRAMESET  G1 probe: an RGB click mapped onto the thermal frame.
%
%   inspect_frameset(projectDir)              click on the RGB image
%   inspect_frameset(projectDir, 0.45, 0.42)  scripted click (normalized)
%
%   Same identity registration as epr inspect: the board is assumed to fill
%   the same normalized rectangle in RGB and thermal. G3 replaces that.

    arguments
        projectDir (1, 1) string
        x (1, 1) double = NaN
        y (1, 1) double = NaN
        frameSetId (1, 1) double = NaN
    end

    set = load_frameset(projectDir, frameSetId);

    if isnan(x) || isnan(y)
        figure("Name", "EPR RGB — click a point");
        image(set.rgb);
        axis image off;
        title(sprintf("frame set %d — click RGB", set.frame_set_id));
        [cx, cy] = ginput(1);
        x = cx / size(set.rgb, 2);
        y = cy / size(set.rgb, 1);
        x = min(max(x, 0), 1);
        y = min(max(y, 0), 1);
    end

    probe = probe_point(set, x, y);
    print_probe(set, probe);

    figure("Name", "EPR thermal");
    imagesc(set.celsius);
    axis image;
    colormap("hot");
    colorbar;
    hold on;
    plot(probe.thermal_pixel(1), probe.thermal_pixel(2), "w+", "MarkerSize", 16, "LineWidth", 2);
    title(sprintf("T = %.2f C", probe.temperature_c));
end

function probe = probe_point(set, x, y)
    % Identity in normalized space — keep in lockstep with
    % epr.processing.registration.point_to_pixel (0-based, truncated).
    [rgbH, rgbW, ~] = size(set.rgb);
    [thH, thW] = size(set.celsius);
    rgbPx = [to_pixel(x, rgbW), to_pixel(y, rgbH)];
    thPx = [to_pixel(x, thW), to_pixel(y, thH)];

    half = 0.025;
    x0 = min(max(0, x - half), 1 - 1e-6);
    y0 = min(max(0, y - half), 1 - 1e-6);
    x1 = min(x0 + 0.05, 1);
    y1 = min(y0 + 0.05, 1);
    c1 = to_pixel(x0, thW);
    r1 = to_pixel(y0, thH);
    c2 = to_pixel(x1, thW);
    r2 = to_pixel(y1, thH);
    patch = set.celsius(r1:r2, c1:c2);

    probe.rgb_xy = [x, y];
    probe.thermal_xy = [x, y];
    probe.rgb_pixel = rgbPx;
    probe.thermal_pixel = thPx;
    probe.temperature_c = set.celsius(thPx(2), thPx(1));
    probe.region_mean_c = mean(patch, "all");
    probe.region_max_c = max(patch, [], "all");
end

function px = to_pixel(normalized, extent)
    % 1-based equivalent of Python min(int(n * extent), extent - 1).
    px = min(floor(normalized * extent) + 1, extent);
end

function print_probe(set, probe)
    ambient = set.metadata.environment.ambient_temperature_c;
    distance = set.metadata.geometry.distance_mm;
    fprintf("frame set %d  click %.3f,%.3f\n", set.frame_set_id, probe.rgb_xy(1), probe.rgb_xy(2));
    fprintf("  RGB pixel      (%d, %d) of %dx%d\n", ...
        probe.rgb_pixel(1) - 1, probe.rgb_pixel(2) - 1, size(set.rgb, 2), size(set.rgb, 1));
    fprintf("  thermal pixel  (%d, %d) of %dx%d\n", ...
        probe.thermal_pixel(1) - 1, probe.thermal_pixel(2) - 1, size(set.celsius, 2), size(set.celsius, 1));
    fprintf("  T              %.2f C  (region mean %.2f  max %.2f)\n", ...
        probe.temperature_c, probe.region_mean_c, probe.region_max_c);
    fprintf("  ambient        %.2f C  distance %.1f mm\n", ambient, distance);
end
