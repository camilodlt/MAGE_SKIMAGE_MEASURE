module MAGE_SKIMAGE_MEASURE
using Pkg
using PythonCall
using DistributedNext
using UTCGP
using UTCGP: BinaryPixel, FunctionBundle
import FixedPointNumbers
using TimerOutputs
using Statistics
import ImageMorphology: label_components
import Base.Threads as Threads

const to = TimerOutput()
const N0f8 = FixedPointNumbers.N0f8
const wrap = PythonCall.pynew() # measure module
const sys = PythonCall.pynew()
dir = @__DIR__
home = dirname(dir)
const SE = trues(3, 3)

function gc()
    return PythonCall.GIL.lock() do
        GC.gc(true)
        PythonCall.GC.gc()
        GC.gc()
    end
end

function __init__()
    global wrap, sys
    # println("Adding to python sys : ", home)
    PythonCall.pycopy!(sys, PythonCall.pyimport("sys"))
    sys.path.insert(0, home)
    return PythonCall.pycopy!(wrap, PythonCall.pyimport("wrap"))
end

bundle_float_skimagemeasure = FunctionBundle(UTCGP.float_caster, () -> 0.0)

# API

function _extract_params(I)
    TT = Base.unwrap_unionall(I).parameters[2]
    UTCGP._validate_factory_type(TT)
    s1 = I.parameters[1].parameters[1]
    s2 = I.parameters[1].parameters[2]
    StorageType = TT.types[1] # UInt8, UInt16 ...
    return TT, s1, s2, StorageType
end

function nparray_to_op(res, operation)::Float64
    return operation(PyArray{Float64}(res; copy = false))
end

function label(img, SE = SE)
    return label_components(img, SE)
end

# --- PRIVATE A
function _area(img)
    l = label(img)
    return wrap.area(l)
end
function _area_bbox(img)
    l = label(img)
    return wrap.area_bbox(l)
end
function _area_convex(img)
    l = label(img)
    return wrap.area_convex(l)
end
function _area_filled(img)
    l = label(img)
    return wrap.area_filled(l)
end
function _axis_major_length(img)
    l = label(img)
    return wrap.axis_major_length(l)
end
function _axis_minor_length(img)
    l = label(img)
    return wrap.axis_minor_length(l)
end

# --- PRIVATE E
function _eccentricity(img)
    l = label(img)
    return wrap.eccentricity(l)
end
function _equivalent_diameter_area(img)
    l = label(img)
    return wrap.equivalent_diameter_area(l)
end
function _euler_number(img)
    l = label(img)
    return wrap.euler_number(l)
end
function _extent(img)
    l = label(img)
    return wrap.extent(l)
end

# --- PRIVATE F
function _feret_diameter_max(img)
    l = label(img)
    return wrap.feret_diameter_max(l)
end

# --- PRIVATE I
# function _intensity_max(img, th, tmp_th_)
#     l = _threshold_and_label(img, th, tmp_th_)
#     wrap.intensity_max(l)
# end
# function _intensity_mean(img, th, tmp_th_)
#     l = _threshold_and_label(img, th, tmp_th_)
#     wrap.intensity_mean(l)
# end
# function _intensity_min(img, th, tmp_th_)
#     l = _threshold_and_label(img, th, tmp_th_)
#     wrap.intensity_min(l)
# end
# function _intensity_std(img, th, tmp_th_)
#     l = _threshold_and_label(img, th, tmp_th_)
#     wrap.intensity_std(l)
# end

# --- PRIVATE N
function _num_pixels(img)
    l = label(img)
    return wrap.num_pixels(l)
end

# --- PRIVATE O
function _orientation(img)
    l = label(img)
    return wrap.orientation(l)
end

# --- PRIVATE P
function _perimeter(img)
    l = label(img)
    return wrap.perimeter(l)
end
function _perimeter_crofton(img)
    l = label(img)
    return wrap.perimeter_crofton(l)
end

# --- PRIVATE S
function _solidity(img)
    l = label(img)
    return wrap.solidity(l)
end

operations = [mean, sum, median, minimum, maximum]

# function run_op(img, operation, operation_reduction)
#     r::Float64 = 0.0
#     res = operation(img)
#     r += nparray_to_op(res, operation_reduction)
#     @info "running python op in thread $(threadid()) in process $(myid())"
#     return r
# end

function run_op(img, operation, operation_reduction)
    return PythonCall.GIL.lock() do
        # @info "running python op in thread $(Threads.threadid()) in process $(myid())"
        r::Float64 = 0.0
        res = operation(img)
        r += nparray_to_op(res, operation_reduction)
        return r
    end
end
# --- PUBLIC API A

# --- --- AREA
for op in operations
    fn_name = "$(op)_area"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)

        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _area, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end


# --- --- AREA BBOX
for op in operations
    fn_name = "$(op)_areabbox"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _area_bbox, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end

# --- --- AREA CONVEX
for op in operations
    fn_name = "$(op)_areaconvex"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _area_convex, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end


# --- --- AREA FILLED
for op in operations
    fn_name = "$(op)_areafilled"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _area_filled, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end

# --- --- AXIS MAJOR LENGTH
for op in operations
    fn_name = "$(op)_axis_major_length"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _axis_major_length, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end


# --- --- AXIS MINOR LENGTH
for op in operations
    fn_name = "$(op)_axis_minor_length"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _axis_minor_length, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end

# --- PUBLIC API E

# --- --- ECCENTRICITY
for op in operations
    fn_name = "$(op)_eccentricity"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _eccentricity, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end


# --- --- EQUIVALENT DIAMETER AREA
for op in operations
    fn_name = "$(op)_equivalent_diameter_area"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _equivalent_diameter_area, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end

# --- --- EULER NUMBER
for op in operations
    fn_name = "$(op)_euler_number"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _euler_number, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end


# --- --- EXTENT
for op in operations
    fn_name = "$(op)_extent"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _extent, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end

# --- PUBLIC API F

# --- --- FERET DIAMETER MAX
for op in operations
    fn_name = "$(op)_feret_diameter_max"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _feret_diameter_max, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end

# --- PUBLIC API I

# --- --- INTENSITY MAX
# for op in operations
#     fn_name = "$(op)_intensity_max"
#     fn_name_factory = Symbol("$(fn_name)_factory")

#     f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
#         tmp = fn_name
#         new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
#         TT, s1, s2, StorageType = _extract_params(I)
#         f = @eval function $new_fn_name(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
#             res = _intensity_max(img)
#             return nparray_to_op(res, $op)
#         end
#         return f
#     end

#     UTCGP.append_method!(
#         bundle_float_skimagemeasure,
#         f,
#         fn_name_factory,
#     )
# end

# --- --- INTENSITY MEAN
# for op in operations
#     fn_name = "$(op)_intensity_mean"
#     fn_name_factory = Symbol("$(fn_name)_factory")

#     f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
#         tmp = fn_name
#         new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
#         TT, s1, s2, StorageType = _extract_params(I)
#         f = @eval function $new_fn_name(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
#             res = _intensity_mean(img)
#             return nparray_to_op(res, $op)
#         end
#         return f
#     end

#     UTCGP.append_method!(
#         bundle_float_skimagemeasure,
#         f,
#         fn_name_factory,
#     )
# end


# --- --- INTENSITY MIN
# for op in operations
#     fn_name = "$(op)_intensity_min"
#     fn_name_factory = Symbol("$(fn_name)_factory")

#     f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
#         tmp = fn_name
#         new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
#         TT, s1, s2, StorageType = _extract_params(I)
#         f = @eval function $new_fn_name(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
#             res = _intensity_min(img)
#             return nparray_to_op(res, $op)
#         end
#         return f
#     end

#     UTCGP.append_method!(
#         bundle_float_skimagemeasure,
#         f,
#         fn_name_factory,
#     )
# end

# --- --- INTENSITY STD
# for op in operations
#     fn_name = "$(op)_intensity_std"
#     fn_name_factory = Symbol("$(fn_name)_factory")

#     f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
#         tmp = fn_name
#         new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
#         TT, s1, s2, StorageType = _extract_params(I)
#         f = @eval function $new_fn_name(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
#             res = _intensity_std(img)
#             return nparray_to_op(res, $op)
#         end
#         return f
#     end

#     UTCGP.append_method!(
#         bundle_float_skimagemeasure,
#         f,
#         fn_name_factory,
#     )
# end

# --- PUBLIC API N

# --- --- NUMPIXELS
for op in operations
    fn_name = "$(op)_num_pixels"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _num_pixels, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end

# --- PUBLIC API O

# --- --- ORIENTATION
for op in operations
    fn_name = "$(op)_orientation"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _orientation, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end

# # --- PUBLIC API P

# --- --- PERIMETER
for op in operations
    fn_name = "$(op)_perimeter"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _perimeter, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end


# --- --- PERIMETER CROFTON
for op in operations
    fn_name = "$(op)_perimeter_crofton"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _perimeter_crofton, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end

# --- PUBLIC API S

# --- --- SOLIDITY
for op in operations
    fn_name = "$(op)_solidity"
    fn_name_factory = Symbol("$(fn_name)_factory")

    f = ((i::Type{I}) where {I <: UTCGP.SizedImage2D{S1, S2, T}} where {S1, S2, T <: BinaryPixel}) -> begin
        tmp = fn_name
        new_fn_name = Symbol("$(tmp)_$(S1)_$(S2)_$(T)")
        TT, s1, s2, StorageType = _extract_params(I)
        interpolated_quote = quote
            struct $new_fn_name{T} <: UTCGP.AbstractFunction
                operation::T
            end
            function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
                return remotecall_fetch(run_op, rand(workers()), img, _solidity, callable.operation)
            end
            $new_fn_name{typeof($op)}($op)
        end
        return eval(interpolated_quote)
    end

    UTCGP.append_method!(
        bundle_float_skimagemeasure,
        f,
        fn_name_factory,
    )
end

export bundle_float_skimagemeasure
end

# Turn this into a macro :
# interpolated_quote = quote
#     struct $new_fn_name{T} <: UTCGP.AbstractFunction
#         operation::T
#     end
#     function (callable::$new_fn_name)(img::CONCT, args::Vararg{Any}) where {CONCT <: $I}
#         return remotecall_fetch(run_op, rand(workers()), img, _solidity, callable.operation)
#     end
#     $new_fn_name{typeof($op)}($op)
# end
