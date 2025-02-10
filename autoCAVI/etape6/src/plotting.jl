using Gadfly, Cairo, Fontconfig, Distributions, Mamba, DataFrames

include("cavi.jl");

"""
    plotTraceCAVI(trace, name)

Plot evolution of the KL divergence over CAVI epochs.

# Arguments
- `trace::DenseVector`: Trace of the parameter.
- `name::String`: Name of the parameter.
"""
function plotTraceCAVI(trace::DenseVector, name::String)
    
    set_default_plot_size(15cm ,10cm)

    n_trace = length(trace);

    plot(
        layer(x=1:n_trace, y=trace, Geom.line),
        layer(x=1:n_trace, y=trace, Geom.point, shape=[Shape.cross], Theme(default_color="red")),
        Theme(background_color="white"),
        Guide.title("CAVI trace of $name"),
        Guide.xlabel("Iteration"),
        Guide.ylabel("Value"),
    )
end


"""
    plotXiCorrelation(Mstart, Mend)
"""
function plotXiCorrelation(Mstart::Integer, Mend::Integer)

    x = [];
    corr12 = [];
    corr13 = [];
    corr23 = [];

    nEpoch = 2;
    epochSize = 10;

    for Msize = Mstart:Mend

        println("Grille de taille $Msize x $Msize...")
        
        M₁ = Msize;
        M₂ = Msize;
        M = M₁ * M₂;
        
        push!(x, Msize);

        initialValues = Dict(
            :μ => zeros(M),
            :ϕ => zeros(M),
            :ξ => 0.0,
        );

        sumCorr13 = 0;
        sumCorr12 = 0;
        sumCorr23 = 0;

        n_occ = 10;
        
        for occ = 1:n_occ

            Random.seed!(300 + occ);
            Fmu = iGMRF(M₁, M₂, 1, 10);
            Fphi = iGMRF(M₁, M₂, 1, 100);
            gridTarget = generateTargetGrid(Fmu, Fphi);
            gridTarget[:, :, 1] = gridTarget[:, :, 1] .+ 10.0;
            gridTarget[:, :, 2] = gridTarget[:, :, 2] .+ 1.0;
            gridTarget[:, :, 3] = gridTarget[:, :, 3] .+ .3;
            nobs = 100;
            data = generateData(gridTarget, nobs);
        
            spatialScheme = Dict(
                :M => M,
                :Fmu => Fmu,
                :Fphi => Fphi,
                :data => data,
            );
            
            res = runCAVI(nEpoch, epochSize, initialValues, spatialScheme);
            marginal = res.approxMarginals[1];
            
            cov = Matrix(params(marginal)[2]);
            corr = covToCorr(cov);
    
            sumCorr13 += corr[1, 3];
            sumCorr12 += corr[1, 2];
            sumCorr23 += corr[2, 3];
        end

        push!(corr13, sumCorr13/n_occ);
        push!(corr12, sumCorr12/n_occ);
        push!(corr23, sumCorr23/n_occ);

    end

    data = vcat(
        DataFrame(x=x, y=corr12, curve="corr(μ,ϕ)"),
        DataFrame(x=x, y=corr13, curve="corr(μ,ξ)"),
        DataFrame(x=x, y=corr23, curve="corr(ϕ,ξ)"),
    )

    set_default_plot_size(15cm ,20cm)

    plot(
        data, x=:x, y=:y, color=:curve, Geom.line,
        Theme(background_color="white"),
        Guide.title("Évolution de la corrélation de ξ dans la cellule 1."),
        Guide.xlabel("Taille du côté de la grille"),
        Guide.ylabel("Valeur"),
    )
end