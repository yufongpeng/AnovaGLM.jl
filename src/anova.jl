# ===========================================================================================
# Main API
@doc """
    anova(<models>...; test::Type{<: GoodnessOfFit},  <keyword arguments>)
    anova(test::Type{<: GoodnessOfFit}, <models>...;  <keyword arguments>)

Analysis of variance.

Return `AnovaResult{M, test, N}`

* `models`: model objects
    1. `TableRegressionModel{<: LinearModel}` fitted by `GLM.lm`
    2. `TableRegressionModel{<: GeneralizedLinearModel}` fitted by `GLM.glm`
    If mutiple models are provided, they should be nested and the last one is the most saturated.
* `test`: test statistics for goodness of fit. Available tests are [`LikelihoodRatioTest`] ([`LRT`]) and [`FTest`]. The default is based on the model type.
    1. `TableRegressionModel{<: LinearModel}`: `FTest`.
    2. `TableRegressionModel{<: GeneralizedLinearModel}`: based on distribution function, see `canonicalgoodnessoffit`.

Other keyword arguments:
* When one model is provided:  
    1. `type` specifies type of anova (1, 2 or 3). Default value is 1.
* When multiple models are provided:  
    1. `check`: allows to check if models are nested. Defalut value is true. Some checkers are not implemented now.
    2. `isnested`: true when models are checked as nested (manually or automatically). Defalut value is false. 

For fitting new models and conducting anova at the same time, see [`anova_lm`](@ref) for `LinearModel`, [`anova_glm`](@ref) for `GeneralizedLinearModel`.
"""
anova(::Val{:AnovaGLM})

anova(models::Vararg{TableRegressionModel{<: LinearModel, <: AbstractArray}, N}; 
        test::Type{T} = FTest,
        kwargs...) where {N, T <: GoodnessOfFit} = 
    anova(test, models...; kwargs...)

anova(models::Vararg{TableRegressionModel{<: GeneralizedLinearModel, <: AbstractArray}, N}; 
        test::Type{T} = canonicalgoodnessoffit(models[1].model.rr.d),
        kwargs...) where {N, T <: GoodnessOfFit} = 
    anova(test, models...; kwargs...)

# ==================================================================================================================
# ANOVA by F test 
# LinearModels

anova(::Type{FTest}, 
    trm::TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel{<: GLM.GlmResp{T, <: Normal, IdentityLink}}}}; 
    type::Int = 1,
    kwargs...) where T = _anova_vcov(trm; type, kwargs...)

function _anova_vcov(trm::TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}; 
                    type::Int = 1, kwargs...)
    type in [1, 2, 3] || throw(ArgumentError("Invalid type"))

    assign = trm.mm.assign
    df = dof(assign)
    filter!(>(0), df)
    # May exist some floating point error from dof_residual
    push!(df, round(Int, dof_residual(trm)))
    df = tuple(df...)
    if type in [1, 3] 
        # vcov methods
        varβ = vcov(trm.model)
        β = trm.model.pp.beta0
        if type == 1
            fs = abs2.(cholesky(Hermitian(inv(varβ))).U * β) 
            offset = first(assign) == 1 ? 0 : 1
            fstat = ntuple(last(assign) - offset) do fix
                sum(fs[findall(==(fix + offset), assign)]) / df[fix]
            end
        else
            # calculate block by block
            offset = first(assign) == 1 ? 0 : 1
            fstat = ntuple(last(assign) - offset) do fix
                select = findall(==(fix + offset), assign)
                β[select]' * inv(varβ[select, select]) * β[select] / df[fix]
            end
        end
        σ² = dispersion(trm.model, true)
        devs = (fstat .* σ²..., σ²) .* df
    else
        # refit methods
        devs = deviances(trm; type, kwargs...)
        MSR = devs ./ df
        fstat = MSR[1:end - 1] ./ dispersion(trm.model, true)
    end
    pvalue = (ccdf.(FDist.(df[1:end - 1], last(df)), abs.(fstat))..., NaN)
    AnovaResult{FTest}(trm, type, df, devs, (fstat..., NaN), pvalue, NamedTuple())
end


function anova(::Type{FTest}, 
                trm::TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}; 
                type::Int = 1, kwargs...)
    type in [1, 2, 3] || throw(ArgumentError("Invalid type"))

    assign = trm.mm.assign
    devs = deviances(trm; type, kwargs...)
    df = dof(assign)
    filter!(>(0), df)
    # May exist some floating point error from dof_residual
    push!(df, round(Int, dof_residual(trm)))
    length(df) == length(devs) + 1 && popfirst!(df)
    df = tuple(df...)
    msr = devs ./ df
    fstat = msr[1:end - 1] ./ dispersion(trm.model, true)
    pvalue = (ccdf.(FDist.(df[1:end - 1], last(df)), abs.(fstat))..., NaN)
    AnovaResult{FTest}(trm, type, df, devs, (fstat..., NaN), pvalue, NamedTuple())
end

# ----------------------------------------------------------------------------------------
# ANOVA for genaralized linear models
# λ = -2ln(𝓛(̂θ₀)/𝓛(θ)) ~ χ²ₙ , n = difference of predictors

function anova(::Type{LRT}, 
            trm::TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}})
    Δdev = deviances(trm, type = 1)
    df = dof(trm.mm.assign)
    filter!(>(0), df)
    isnullable(trm.model) || popfirst!(df)
    df = tuple(df...)
    # den = last(ss) / (nobs(trm) - dof(trm) + 1)
    # lrstat = ss[1:end - 1] ./ den
    σ² = dispersion(trm.model, true)
    lrstat = Δdev[1:end - 1] ./ σ²
    n = length(lrstat)
    dev = collect(Δdev)
    i = n
    while i > 0
        dev[i] += dev[i + 1]
        i -= 1
    end
    pval = ccdf.(Chisq.(df), abs.(lrstat))
    AnovaResult{LRT}(trm, 1, df, tuple(dev[2:end]...), lrstat, pval, NamedTuple())
end

# =================================================================================================================
# Nested models 

function anova(::Type{FTest}, 
        trms::Vararg{TableRegressionModel{<: Union{LinearModel, GeneralizedLinearModel}}}; 
        check::Bool = true,
        isnested::Bool = false)   
    df = dof.(trms)
    ord = sortperm(collect(df))
    df = df[ord]
    trms = trms[ord]
    dfr = round.(Int, dof_residual.(trms))
    # May exist some floating point error from dof_residual
    # check comparable and nested
    check && @warn "Could not check whether models are nested: results may not be meaningful"
    ftest_nested(trms, df, dfr, deviance.(trms), dispersion(last(trms).model, true))
end

function anova(::Type{LRT}, 
        trms::Vararg{<: TableRegressionModel}; 
        check::Bool = true,
        isnested::Bool = false)
    df = dof.(trms)
    ord = sortperm(collect(df))
    trms = trms[ord]
    df = df[ord]
    # check comparable and nested
    check && @warn "Could not check whether models are nested: results may not be meaningful"
    lrt_nested(trms, df, deviance.(trms), dispersion(last(trms).model, true))
end


# =================================================================================================================================
# Fit new models

"""
    anova_lm(X, y; test::Type{<: GoodnessOfFit} = FTest, <keyword arguments>) 

    anova_lm(test::Type{<: GoodnessOfFit}, X, y; <keyword arguments>)

    anova(test::Type{<: GoodnessOfFit}, ::Type{LinearModel}, X, y; 
        type::Int = 1, 
        <keyword arguments>)

ANOVA for simple linear regression.

The arguments `X` and `y` can be a `Matrix` and a `Vector` or a `Formula` and a `DataFrame`. 
* `type` specifies type of anova (1, 2 or 3). Default value is 1.
* `dropcollinear` controls whether or not `lm` accepts a model matrix which is less-than-full rank. If true (the default), only the first of each set of linearly-dependent columns is used. The coefficient for redundant linearly dependent columns is 0.0 and all associated statistics are set to NaN.

`anova_lm` generate a `TableRegressionModel` object, which is fitted by `lm`.
"""
anova_lm(X, y; 
        test::Type{<: GoodnessOfFit} = FTest, 
        kwargs...) = 
    anova(test, LinearModel, X, y; kwargs...)

anova_lm(test::Type{<: GoodnessOfFit}, X, y; kwargs...) = 
    anova(test, LinearModel, X, y; kwargs...)

function anova(test::Type{<: GoodnessOfFit}, ::Type{LinearModel}, X, y; 
        type::Int = 1, 
        kwargs...)
    model = lm(X, y; kwargs...)
    anova(test, model; type = type)
end

"""
    anova_glm(X, y, d::UnivariateDistribution, l::Link = canonicallink(d); 
            test::Type{<: GoodnessOfFit} = canonicalgoodnessoffit(d), <keyword arguments>)

    anova_glm(test::Type{<: GoodnessOfFit}, X, y, d::UnivariateDistribution, l::Link = canonicallink(d); <keyword arguments>)

    anova(test::Type{<: GoodnessOfFit}, X, y, d::UnivariateDistribution, l::Link = canonicallink(d); <keyword arguments>)

ANOVA for genaralized linear models.

The arguments `X` and `y` can be a `Matrix` and a `Vector` or a `Formula` and a `DataFrame`. 
* `d`: a `GLM.UnivariateDistribution`.
* `l`: a `GLM.Link`

For other keyword arguments, see `fit`.
"""
anova_glm(X, y, 
        d::UnivariateDistribution, l::Link = canonicallink(d); 
        test::Type{<: GoodnessOfFit} = canonicalgoodnessoffit(d), 
        kwargs...) = 
    anova(test, GeneralizedLinearModel, X, y, d, l; kwargs...)

anova_glm(test::Type{<: GoodnessOfFit}, X, y, 
        d::UnivariateDistribution, l::Link = canonicallink(d); 
        kwargs...) = 
    anova(test, GeneralizedLinearModel, X, y, d, l; kwargs...)

function anova(test::Type{<: GoodnessOfFit}, ::Type{GeneralizedLinearModel}, X, y, 
        d::UnivariateDistribution, l::Link = canonicallink(d);
        type::Int = 1,
        kwargs...)
    trm = glm(X, y, d, l; kwargs...)
    anova(test, trm; type, kwargs... )
end 


"""
    GLM.glm(f::FormulaTerm, df::DataFrame, d::Binomial, l::GLM.Link, args...; kwargs...)

Automatically transform dependent variable into 0/1 for family `Binomial`.
"""
GLM.glm(f::FormulaTerm, df::DataFrame, d::Binomial, l::Link, args...; kwargs...) = 
    fit(GeneralizedLinearModel, f, 
        combine(df, :, f.lhs.sym => ByRow(==(last(unique(df[!, f.lhs.sym])))) => f.lhs.sym), 
        d, l, args...; kwargs...)

