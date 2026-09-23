function set = load_frameset(projectDir, frameSetId)
%LOAD_FRAMESET  Read one raw EPR frame set (PNG + TIFF + JSON).
%
%   set = load_frameset(projectDir)
%   set = load_frameset(projectDir, frameSetId)
%
%   set.rgb       HxWx3 uint8, RGB
%   set.nir       HxW   uint8
%   set.thermal   hxw   uint16, radiometric centikelvin
%   set.celsius   hxw   double, degrees C
%   set.metadata  struct from the JSON sidecar
%
%   Same layout as RawRecorder: <project>/raw/{rgb,nir,thermal,metadata}/

    arguments
        projectDir (1, 1) string
        frameSetId (1, 1) double = NaN
    end

    raw = fullfile(projectDir, "raw");
    if isnan(frameSetId)
        frameSetId = latest_id(raw);
    end

    stem = sprintf("%012d", frameSetId);
    rgbPath = fullfile(raw, "rgb", stem + ".png");
    nirPath = fullfile(raw, "nir", stem + ".png");
    thPath = fullfile(raw, "thermal", stem + ".tiff");
    metaPath = fullfile(raw, "metadata", stem + ".json");

    required = [rgbPath, nirPath, thPath, metaPath];
    missing = required(~isfile(required));
    if ~isempty(missing)
        error("epr:incompleteFrameSet", "missing:\n  %s", strjoin(missing, "\n  "));
    end

    set.rgb = imread(rgbPath);
    set.nir = imread(nirPath);
    set.thermal = imread(thPath);
    if ~isa(set.thermal, "uint16")
        set.thermal = uint16(set.thermal);
    end
    set.celsius = double(set.thermal) / 100 - 273.15;
    set.metadata = jsondecode(fileread(metaPath));
    set.frame_set_id = frameSetId;
end

function id = latest_id(raw)
    listing = dir(fullfile(raw, "metadata", "*.json"));
    if isempty(listing)
        error("epr:emptyProject", "no raw frame sets in %s", raw);
    end
    ids = arrayfun(@(f) str2double(f.name(1:end-5)), listing);
    id = max(ids);
end
