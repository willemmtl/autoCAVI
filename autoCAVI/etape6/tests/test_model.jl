using Test, GMRF, SpecialFunctions, Distributions

include("../src/model.jl");

@testset "model.jl" begin
    
    @testset "logposterior(μ, ϕ, ξ; Fmu, Fphi, data)" begin
        
        M₁ = 2;
        M₂ = 2;
        Fmu = iGMRF(M₁, M₂, 1, 1.0); # Peu importe le kappa
        Fphi = iGMRF(M₁, M₂, 1, 1.0); # Peu importe le kappa
        data = [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ];

        μ = zeros(4);
        ϕ = zeros(4);
        ξ = 0.0;

        @test logposterior(μ, ϕ, ξ, Fmu=Fmu, Fphi=Fphi, data=data) ≈ (
            - 8 - 8 / ℯ 
            + 1.5 * (log(10) + log(100)) 
            + loggamma(15) - loggamma(6) - loggamma(9) + 13 * log(.5)
        )

    end


    @testset "neighborsMean(cellIndex, θ, F)" begin
        
        F = iGMRF(2, 2, 1, 1.0); # Peu importe le kappa
        θ = [1.0, 2.0, 3.0, 5.0];

        @test neighborsMean(1, θ, F) ≈ 2.5;
        @test neighborsMean(2, θ, F) ≈ 3.0;
    end


    @testset "fullconditional(i, θi; μ, ϕ, μ̄i, ϕ̄i, Fmu, Fphi)" begin
        
        M₁ = 2;
        M₂ = 2;
        M = M₁ * M₂;

        i = 1;
        θi = [1.0, 0.0, 0.0];
        μ = zeros(M);
        ϕ = zeros(M);
        μ̄i = 0.0;
        ϕ̄i = 0.0;
        Fmu = iGMRF(M₁, M₂, 1, 10.0);
        Fphi = iGMRF(M₁, M₂, 1, 100.0);
        data = [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ];

        @test logfullconditional(i, θi, μ=μ, ϕ=ϕ, μ̄i=μ̄i, ϕ̄i=ϕ̄i, Fmu=Fmu, Fphi=Fphi, data=data) ≈ (
            - 6 - 6 * exp(-1) # somme des loglike GEV pour les autres cellules
            - 2 # loglike GEV pour la cellule i
            - .5 * log(2*pi / 20) - 10 # Priori de mu
            - .5 * log(2*pi / 200) # Priori de phi
            + loggamma(15) - loggamma(6) - loggamma(9) + 13 * log(.5) # Priori de xi
        )
    end
end