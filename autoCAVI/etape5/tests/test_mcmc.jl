using Test, GMRF, Distributions

include("../src//mcmc.jl");

@testset "mcmc.jl" begin
    
    @testset "fullconditionalIGMRF(F, θ)" begin

        F = iGMRF(2, 2, 1, 1.0);
        θ = [0.0, 2.0, 0.0, 0.0];

        pds = fullconditionalsIGMRF(F, θ);

        @test pds[1] == NormalCanon(2.0, 2.0);
        
    end


    @testset "vertices_update!(θ, F, θ̃, logL, ind)" begin
        
        M₁ = 3;
        M₂ = 3;
        M = M₁ * M₂;

        F = iGMRF(M₁, M₂, 1, 1.0);
        θ = zeros(M);
        θ̃ = ones(M);
        logL = [Inf for i = 1:M];
        ind = [1, 3, 5, 7, 9];

        vertices_update!(θ, F, θ̃, logL, ind);

        @test sum(θ) ≈ 5.0;
        @test θ[1] == 1.0;
        @test θ[2] == 0.0;

    end


    @testset "vertices_update!(θ, F, θ̃, logL)" begin
        
        M₁ = 3;
        M₂ = 3;
        M = M₁ * M₂;

        F = iGMRF(M₁, M₂, 1, 1.0);
        θ = zeros(M);
        θ̃ = ones(M);
        logL = [Inf for i = 1:M];

        θ = vertices_update(θ, F, θ̃, logL);

        @test sum(θ) ≈ 9.0;

    end


    @testset "createChain(M, μ, ϕ, ξ)" begin
        
        M = 9;
        μ = zeros(M, 2);
        ϕ = zeros(M, 2);
        ξ = zeros(M, 2);

        μ[:, 2] = [i for i = 1:M];
        ϕ[:, 2] = [i for i = M+1:2*M];
        ξ[:, 2] = [i for i = 2*M+1:3*M];

        chain = createChain(M, μ, ϕ, ξ);

        @test chain[:, "ξ2", 1].value[2] ≈ 20.0;

    end


    @testset "mcmc(datastructure, niter, initialvalues, stepsize)" begin
        
        M₁ = 3;
        M₂ = 3;
        M = M₁ * M₂;

        datastructure = Dict(
            :Y => [
                [1.0, 1.0] for i = 1:M
            ],
            :Fmu => iGMRF(M₁, M₂, 1, 10.0),
            :Fphi => iGMRF(M₁, M₂, 1, 100.0),
            :Fxi => iGMRF(M₁, M₂, 1, 100000.0),
        );

        niter = 10;

        initialvalues = Dict(
            :μ => zeros(M),
            :ϕ => zeros(M),
            :ξ => zeros(M),
        );

        stepsize = Dict(
            :μ => 1.0,
            :ϕ => 1.0,
            :ξ => 1.0,
        );

        chain = mcmc(datastructure, niter, initialvalues, stepsize);

        @test length(chain.names) == 3 * M;
        @test length(chain[:, "μ1", 1].value) == niter;

    end

end