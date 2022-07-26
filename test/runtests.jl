using AnovaGLM, CSV, RDatasets, DataFrames, CategoricalArrays
using Test
import Base.isapprox

test_show(x) = show(IOBuffer(), x)
macro test_error(x)
    return quote
        try 
            $x
            false
        catch e
            @error e
            true
        end
    end
end

const anova_datadir = joinpath(dirname(@__FILE__), "..", "data")

"Data from R datasets"
iris = dataset("datasets", "iris")

"Data source: https://www.qogdata.pol.gu.se/dataarchive/qog_bas_cs_jan18.dta"
qog18 = CSV.read(joinpath(anova_datadir, "qog18.csv"), DataFrame)

"Data from R package MASS"
quine = dataset("MASS", "quine")

"Data from R datasets"
mtcars = dataset("datasets", "mtcars")
transform!(mtcars, [:VS, :Model] .=> categorical, renamecols = false)

"Data source: https://www.cambridge.org/core/journals/astin-bulletin-journal-of-the-iaa/article/two-studies-in-automobile-insurance-ratemaking/1A169017AE99FCFE6E5F6BF56BCCA9C2"
insurance = CSV.read(joinpath(anova_datadir, "automobile_insurance.csv"), DataFrame)
transform!(insurance, [1, 2] .=> categorical, renamecols = false)

"Data source: https://stats.idre.ucla.edu/stat/stata/dae/poisson_sim"
sim = CSV.read(joinpath(anova_datadir, "poisson_sim.csv"), DataFrame)
transform!(sim, [:id, :prog] .=> categorical, renamecols = false)

# custimized approx
isapprox(x::NTuple{N, Float64}, y::NTuple{N, Float64}, atol::NTuple{N, Float64} = x ./ 1000) where N = 
    all(map((a, b, c)->isapprox(a, b, atol = c > eps(Float64) ? c : eps(Float64)), x, y, atol))

@testset "AnovaGLM.jl" begin
    @testset "LinearModel" begin 
        @testset "Simple linear regression" begin
            lm0, lm1, lm2, lm3, lm4 = nestedmodels(LinearModel, @formula(SepalLength ~ SepalWidth * Species), iris, dropcollinear = false)
            global aov3 = anova(lm4, type = 3)
            global aov2 = anova_lm(@formula(SepalLength ~ SepalWidth * Species), iris, type = 2)
            global aov1 = anova(lm4)
            global aovf = anova(lm0, lm1, lm2, lm3, lm4)
            global aovlr = anova(LRT, lm0, lm1, lm2, lm3, lm4)
            global aov1lr = anova(LRT, lm4)
            global aovlf = anova_lm(FTest, @formula(wdi_lifexp ~ log(gle_rgdpc) * ti_cpi), qog18, type = 2)
            ft = ftest(lm1.model, lm2.model, lm3.model, lm4.model)
            @test !(@test_error(test_show(aov1)))
            @test !(@test_error(test_show(aovf)))
            @test !(@test_error(test_show(aov1lr)))
            @test !(@test_error(test_show(aovlr)))
            @test !(@test_error(test_show(aovlf)))
            @test anova_type(aov1) == 1
            @test nobs(aov2) == ft.nobs
            @test dof(aov3) == (1, 1, 2, 2, 144)
            @test dof_residual(aov1) == 144
            @test isapprox(deviance(aovf)[2:end], ft.ssr)
            @test isapprox(deviance(aov1)[1:end - 1], AnovaBase._diffn(deviance(aovf)))
            @test isapprox(filter(!isnan, teststat(aov1)), filter(!isnan, teststat(aovf)))
            @test isapprox(filter(!isnan, teststat(aov2)), (26485.300978452644, 56.637879295914324, 188.10911669921464, 0.4064420847481629))
            @test isapprox(filter(!isnan, teststat(aov1lr)), filter(!isnan, teststat(aovlr)))
            @test isapprox(coef(lm4), coef(aov3.model))
            @test coefnames(lm4, Val(:anova)) ==  ["(Intercept)", "SepalWidth", "Species", "SepalWidth & Species"]
            @test collect(coefnames(formula(lm1), Val(:anova))) == coefnames((formula(lm1).lhs, formula(lm1).rhs), Val(:anova))
    
        end
    
        @testset "Linear regression with frequency weights" begin
            wlm1, wlm2 = nestedmodels(LinearModel, @formula(Cost / Claims ~ Insured + Merit), insurance, 
                                    wts = insurance.Claims ./ sum(insurance.Claims) .* length(insurance.Claims))
            global aov = anova(wlm2)
            global aovf = anova(wlm1, wlm2)
            @test !(@test_error test_show(aov))
            @test !(@test_error test_show(aovf))
            @test nobs(aov) == sum(aov.model.model.rr.wts)
            @test dof(aov) == (1, 1, 18)
            @test isapprox(filter(!isnan, teststat(aov))[2:end], filter(!isnan, teststat(aovf)))
        end
    end
    
    @testset "GeneralizedLinearModel" begin
        @testset "Gamma regression" begin
            global aov = anova_glm(FTest, @formula(Cost / Claims ~ 0 + Insured + Merit), insurance, Gamma(), 
                                    wts = insurance.Claims ./ sum(insurance.Claims) .* length(insurance.Claims), type = 2)
            @test !(@test_error test_show(aov))
            @test nobs(aov) == nobs(aov.model)
            @test dof(aov) == (1, 4, 15)
            @test isapprox(filter(!isnan, deviance(aov)), AnovaGLM.deviances(aov.model, type = anova_type(aov)))
            @test isapprox(filter(!isnan, teststat(aov)), (6.163653078060339, 2802.3252386290533))
            @test isapprox(filter(!isnan, pval(aov)), (0.025357816283854216, 2.3626325930920802e-21))
        end
    
        @testset "NegativeBinomial regression" begin
            global aov = anova_glm(@formula(Days ~ Eth + Sex + Age + Lrn), quine, NegativeBinomial(2.0), LogLink(), type = 3)
            @test !(@test_error test_show(aov))
            @test nobs(aov) == nobs(aov.model)
            @test dof(aov) == (1, 1, 1, 3, 1, 139)
            @test anova_test(aov) == FTest
            @test isapprox(deviance(aov), AnovaGLM.deviances(aov.model, type = 3))
            @test isapprox(filter(!isnan, teststat(aov)), (227.97422313423752, 13.180680587112887, 0.2840882132754838, 4.037856229143672, 2.6138558930314595))
            @test isapprox(filter(!isnan, pval(aov)), (4.251334236308285e-31, 0.00039667906264309605, 0.5948852219093326, 0.008659894572621351, 0.10820118567663468))
            @test coefnames(aov.model, Val(:anova)) == ["(Intercept)", "Eth", "Sex", "Age", "Lrn"]
        end
    
        @testset "Poisson regression" begin
            gmp = nestedmodels(GeneralizedLinearModel, @formula(num_awards ~ prog * math), sim, Poisson())
            global aov = anova(gmp...)
            lr = lrtest(gmp[2:end]...)
            @test !(@test_error test_show(aov))
            @test first(nobs(aov)) == lr.nobs
            @test dof(aov)[2:end] == lr.dof
            @test anova_test(aov) == LRT
            @test isapprox(deviance(aov)[2:end], lr.deviance)
            @test isapprox(filter(!isnan, pval(aov))[2:end], filter(!isnan, lr.pval))
        end
    
        @testset "Logit regression" begin
            gml = glm(@formula(AM ~ Cyl + HP + WT), mtcars, Binomial(), LogitLink())
            global aov = anova(gml)
            lr = lrtest(nestedmodels(gml)[2:end]...)
            @test !(@test_error test_show(aov))
            @test nobs(aov) == lr.nobs
            @test dof(aov)[2:end] == AnovaBase._diff(lr.dof)
            @test isapprox(deviance(aov), lr.deviance)
            @test isapprox(AnovaGLM.deviances(aov.model)[2:end], AnovaBase._diffn((deviance(aov)..., 0.0)))
            @test isapprox(filter(!isnan, pval(aov))[2:end], filter(!isnan, lr.pval))
        end
    
        @testset "Probit regression" begin
            gmp0 = glm(@formula(AM ~ 1), mtcars, Binomial(), ProbitLink())
            gmp1 = glm(@formula(AM ~ Cyl + HP + WT), mtcars, Binomial(), ProbitLink())
            gmp2 = glm(@formula(AM ~ Cyl * HP * WT), mtcars, Binomial(), ProbitLink())
            global aov = anova(gmp0, gmp1, gmp2)
            lr = lrtest(gmp0, gmp1, gmp2)
            @test !(@test_error test_show(aov))
            @test first(nobs(aov)) == lr.nobs
            @test dof(aov) == lr.dof
            @test isapprox(deviance(aov), lr.deviance)
            @test isapprox(filter(!isnan, pval(aov)), filter(!isnan, lr.pval))
        end
    
        @testset "InverseGaussian regression" begin
            gmi = glm(@formula(SepalLength ~ SepalWidth * Species), iris, InverseGaussian())
            global aov = anova(gmi)
            @test !(@test_error test_show(aov))
            @test nobs(aov) == nobs(gmi)
            @test dof(aov) == (1, 2, 2, 144)
            @test isapprox(filter(!isnan, teststat(aov)), (8.172100334461327, 217.34941014323272, 1.8933247444892272))
            @test isapprox(filter(!isnan, pval(aov)), (0.004885873691352542, 3.202701170052312e-44, 0.15429963193830074))
        end
    end
end
