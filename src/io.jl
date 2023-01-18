# ======================================================================================================
# IO
# anovatable api
function anovatable(aov::AnovaResult{<: FullModel{<: TableRegressionModel{<: LinearModel}, N}, FTest}; rownames = push!(prednames(aov), "(Residuals)")
    ) where N
    dfr = round(Int, dof_residual(aov.anovamodel.model))
    σ² = dispersion(aov.anovamodel.model.model, true)
    AnovaTable([
                [dof(aov)..., dfr], 
                [deviance(aov)..., dfr * σ²], 
                [(deviance(aov) ./ dof(aov))..., σ²], 
                [teststat(aov)..., NaN], 
                [pval(aov)..., NaN]
                ],
              ["DOF", "Exp.SS", "Mean Square", "F value","Pr(>|F|)"],
              rownames, 5, 4)
end 

function anovatable(aov::AnovaResult{<: FullModel{<: TableRegressionModel{<: GeneralizedLinearModel}, N}, FTest}; rownames = push!(prednames(aov), "(Residuals)")) where N
    dfr = round(Int, dof_residual(aov.anovamodel.model.model))
    σ² = dispersion(aov.anovamodel.model.model, true)
    AnovaTable([
                [dof(aov)..., dfr], 
                [deviance(aov)..., dfr * σ²], 
                [(deviance(aov) ./ dof(aov))..., σ²], 
                [teststat(aov)..., NaN], 
                [pval(aov)..., NaN]
                ],
              ["DOF", "ΔDeviance", "Mean ΔDev", "F value","Pr(>|F|)"],
              rownames, 5, 4)
end 

function anovatable(aov::AnovaResult{<: FullModel{<: TableRegressionModel{<: LinearModel}}, LRT}; rownames = prednames(aov))
    AnovaTable(hcat(vectorize.((dof(aov), deviance(aov), teststat(aov), pval(aov)))...),
              ["DOF", "Res.SS", "χ²", "Pr(>|χ²|)"],
              rownames, 4, 3)
end 

function anovatable(aov::AnovaResult{<: FullModel{<: TableRegressionModel{<: GeneralizedLinearModel}}, LRT}; rownames = prednames(aov))
    AnovaTable(hcat(vectorize.((dof(aov), deviance(aov), teststat(aov), pval(aov)))...),
              ["DOF", "Deviance", "χ²", "Pr(>|χ²|)"],
              rownames, 4, 3)
end

function anovatable(aov::AnovaResult{NestedModels{<: TableRegressionModel{<: LinearModel}, N}, FTest}; 
                    rownames = "x" .* string.(1:N)) where N

    rs = r2.(aov.anovamodel.model)
    Δrs = _diff(rs)
    AnovaTable([
                    dof(aov), 
                    [NaN, _diff(dof(aov))...], 
                    round(Int, dof_residual(aov.anovamodel.model)), 
                    rs,
                    [NaN, Δrs...],
                    deviance(aov), 
                    [NaN, _diffn(deviance(aov))...], 
                    teststat(aov), 
                    pval(aov)
                ],
              ["DOF", "ΔDOF", "Res.DOF", "R²", "ΔR²", "Res.SS", "Exp.SS", "F value", "Pr(>|F|)"],
              rownames, 9, 8)
end 

function anovatable(aov::AnovaResult{NestedModels{<: TableRegressionModel{<: LinearModel}, N}, LRT}; 
                    rownames = "x" .* string.(1:N)) where N

    rs = r2.(aov.anovamodel.model)
    Δrs = _diff(rs)
    AnovaTable([
                    dof(aov), 
                    [NaN, _diff(dof(aov))...], 
                    round(Int, dof_residual(aov.anovamodel.model)), 
                    rs,
                    [NaN, Δrs...],
                    deviance(aov), 
                    [NaN, _diffn(deviance(aov))...], 
                    teststat(aov), 
                    pval(aov)
                ],
              ["DOF", "ΔDOF", "Res.DOF", "R²", "ΔR²", "Res.SS", "χ²", "Pr(>|χ²|)"],
              rownames, 8, 7)
end 