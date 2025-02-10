module MAGE_SKIMAGE_MEASURE
using Pkg
using PythonCall
using Distributed
using UTCGP
import FixedPointNumbers
using TimerOutputs
using Statistics

const to = TimerOutput()
const N0f8 = FixedPointNumbers.N0f8
const wrap = PythonCall.pynew() # measure module
dir = @__DIR__
home = dirname(dir)

function __init__()
    println("Adding to python sys : ", home)
    sys = PythonCall.pyimport("sys")
    sys.path.insert(0, home)
    PythonCall.pycopy!(wrap, PythonCall.pyimport("wrap"))
end

bundle_float_skimagemeasure = FunctionBundle(UTCGP.float_caster, () -> 0.0)

# API 
function _th!(img, th, dst)
    th = clamp(th, 0.0, 1.0)
    # dst .= (reinterpret.(true_type, img.img) .> reinterpret(fixed_type(th)))
    dst .= (img.img .> th)
    nothing
end

function _extract_params(I)
    TT = Base.unwrap_unionall(I).parameters[2]
    UTCGP._validate_factory_type(TT)
    s1 = I.parameters[1].parameters[1]
    s2 = I.parameters[1].parameters[2]
    StorageType = TT.types[1] # UInt8, UInt16 ...
    TT, s1, s2, StorageType
end

function _make_tmp_uint_matrix(storage_type, s1, s2)
    Matrix{storage_type}(undef, (s1, s2))
end

function nparray_to_mean(res)
    Statistics.mean(PyArray(res; copy=false))
end

# --- PRIVATE A
function _area(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.area(tmp_th_)
end
function _area_bbox(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.area_bbox(tmp_th_)
end
function _area_convex(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.area_convex(tmp_th_)
end
function _area_filled(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.area_filled(tmp_th_)
end
function _axis_major_length(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.axis_major_length(tmp_th_)
end
function _axis_minor_length(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.axis_minor_length(tmp_th_)
end


# --- PRIVATE E
function _eccentricity(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.eccentricity(tmp_th_)
end
function _equivalent_diameter_area(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.equivalent_diameter_area(tmp_th_)
end
function _euler_number(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.euler_number(tmp_th_)
end
function _extent(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.extent(tmp_th_)
end

# --- PRIVATE F
function _feret_diameter_max(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.feret_diameter_max(tmp_th_)
end

# --- PRIVATE I
function _intensity_max(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.intensity_max(tmp_th_)
end
function _intensity_mean(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.intensity_mean(tmp_th_)
end
function _intensity_min(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.intensity_min(tmp_th_)
end
function _intensity_std(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.intensity_std(tmp_th_)
end

# --- PRIVATE N
function _num_pixels(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.num_pixels(tmp_th_)
end

# --- PRIVATE O
function _orientation(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.orientation(tmp_th_)
end

# --- PRIVATE P
function _perimeter(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.perimeter(tmp_th_)
end
function _perimeter_crofton(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.perimeter_crofton(tmp_th_)
end

# --- PRIVATE S
function _solidity(img, th, tmp_th_)
    _th!(img, th, tmp_th_)
    wrap.solidity(tmp_th_)
end

# --- PUBLIC API A 
function mean_area_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _area(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _area(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_area)
end

function mean_areabbox_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _area_bbox(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _area_bbox(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_area_bbox)
end

function mean_areaconvex_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _area_convex(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _area_convex(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_area_convex)
end

function mean_areafilled_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _area_filled(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _area_filled(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_area_filled)
end

function mean_axismajorlength_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _axis_major_length(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _axis_major_length(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_axis_major_length)
end

function mean_axisminorlength_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _axis_minor_length(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _axis_minor_length(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_axis_minor_length)
end

# --- PUBLIC API E
function mean_eccentricity_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _eccentricity(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _eccentricity(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_eccentricity)
end

function mean_equivalentDiameterArea_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _equivalent_diameter_area(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _equivalent_diameter_area(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_equivalent_diameter_area)
end

function mean_eulerNumber_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _euler_number(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _euler_number(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_euler_number)
end

function mean_extent_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _extent(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _extent(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_extent)
end


# --- PUBLIC API F

function mean_feretDiameterMax_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _feret_diameter_max(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _feret_diameter_max(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_feret_diameter_max)
end

# --- PUBLIC API I

function mean_intensityMax_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _intensity_max(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _intensity_max(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_intensity_max)
end

function mean_intensityMean_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _intensity_mean(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _intensity_mean(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_intensity_mean)
end

function mean_intensityMin_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _intensity_min(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _intensity_min(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_intensity_min)
end

function mean_intensityStd_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _intensity_std(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _intensity_std(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_intensity_std)
end

# --- PUBLIC API N

function mean_numPixels_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _num_pixels(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _num_pixels(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_num_pixels)
end

# --- PUBLIC API O

function mean_orientation_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _orientation(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _orientation(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_orientation)
end

# --- PUBLIC API P

function mean_perimeter_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _perimeter(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _perimeter(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_perimeter)
end

function mean_perimeterCrofton_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _perimeter_crofton(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _perimeter_crofton(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_perimeter_crofton)
end

# --- PUBLIC API S

function mean_solidity_factory(i::Type{I}) where {I<:UTCGP.SizedImage2D{S1,S2,T}} where {S1,S2,T<:N0f8}
    TT, s1, s2, StorageType = _extract_params(I)
    m1 = @eval ((img::CONCT, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _solidity(img, 0.5, tmp_th_)
        nparray_to_mean(res)
    end
    m2 = @eval ((img::CONCT, th::Float64, args::Vararg{Any}) where {CONCT<:$I}) -> begin
        tmp_th_ = _make_tmp_uint_matrix($StorageType, $s1, $s2)
        res = _solidity(img, th, tmp_th_)
        nparray_to_mean(res)
    end
    return ManualDispatcher((m2, m1), :mean_solidity)
end

# BUNDLE 

# A
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_area_factory,
    :mean_area_factory,
)
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_areabbox_factory,
    :mean_areabbox_factory,
)
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_areaconvex_factory,
    :mean_areaconvex_factory,
)
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_areafilled_factory,
    :mean_areafilled_factory,
)
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_axismajorlength_factory,
    :mean_axismajorlength_factory,
)
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_axisminorlength_factory,
    :mean_axisminorlength_factory,
)


# E
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_eccentricity_factory,
    :mean_eccentricity_factory,
)
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_equivalentDiameterArea_factory,
    :mean_equivalentDiameterArea_factory,
)
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_eulerNumber_factory,
    :mean_eulerNumber_factory,
)
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_extent_factory,
    :mean_extent_factory,
)

# F
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_feretDiameterMax_factory,
    :mean_feretDiameterMax_factory,
)

# I

UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_intensityMax_factory,
    :mean_intensityMax_factory,
)
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_intensityMean_factory,
    :mean_intensityMean_factory,
)
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_intensityMin_factory,
    :mean_intensityMin_factory,
)
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_intensityStd_factory,
    :mean_intensityStd_factory,
)

# N 

UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_numPixels_factory,
    :mean_numPixels_factory,
)

# O 
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_orientation_factory,
    :mean_orientation_factory,
)

# P 
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_perimeter_factory,
    :mean_perimeter_factory,
)
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_perimeterCrofton_factory,
    :mean_perimeterCrofton_factory,
)

# S 
UTCGP.append_method!(
    bundle_float_skimagemeasure,
    mean_solidity_factory,
    :mean_solidity_factory,
)

export bundle_float_skimagemeasure
end
