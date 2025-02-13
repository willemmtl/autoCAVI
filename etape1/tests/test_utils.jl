using Test, Optim, Distributions
using Distributions:loglikelihood

include("../src/utils.jl");

@testset "utils.jl" begin
    
    @testset "findMode(f, x0)" begin
        
        f(x::DenseVector) = -(x[1]-1)^2 + 7;

        mode = findMode(f, [0.0]);

        @test mode[1] ≈ 1.0;

        data = [2.0];
        logposterior(θ::DenseVector) = loglikelihood(GeneralizedExtremeValue(θ[1], exp(θ[2]), θ[3]), data);
        
        mode = findMode(θ -> logposterior([0.0, 0.0, θ[3]]), [0.0, 0.0, 0.0])

        @test (mode[1] - 0.0) < .1;
        @test (mode[2] - 0.0) < .1;
        @test mode[3] != 0.0;

    end


    @testset "getSecondDerivative(f, x0)" begin
        
        f(x::Real) = x^2;

        @test getSecondDerivative(f, 12.5) ≈ 2.0;

    end


    @testset "evaluateLogMvDensity(f, supp)" begin

        supp = [1 2; 3 4];
        
        f(x::DenseVector) = x[1] + x[2]^2;

        res = evaluateLogMvDensity(f, supp);

        @test res[1] == 10;
        @test res[2] == 18;

    end


    @testset "generateApproxSample(approxMarginals, N)" begin
        
        marginal1 = Normal(0, 1);
        marginal2 = Normal(1, 1);
        marginal3 = Normal(2, 1);

        approxMarginals = [marginal1, marginal2, marginal3];

        supp = generateApproxSample(approxMarginals, 1000);

        @test size(supp) == (3, 1000);
        @test (mean(supp[1, :]) - 0.0) < .1;
        @test (mean(supp[2, :]) - 1.0) < .1;
        @test (mean(supp[3, :]) - 2.0) < .1;

    end

end