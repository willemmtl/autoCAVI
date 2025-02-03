using Test, Distributions, LinearAlgebra

include("../src/monteCarlo.jl");

@testset "monteCarlo.jl" begin
    
    @testset "generateApproxSample(approxMarginals, N)" begin
        
        approxMarginals = [
            MvNormal(fill(-100, 3), I),
            MvNormal(fill(100, 3), I),
        ];

        supp = generateApproxSample(approxMarginals, 10);

        @test size(supp) == (6, 10);
    
    end


    @testset "evaluateLogMvDensity(f, supp)" begin

        supp = [1 2; 3 4];
        
        f(x::DenseVector) = x[1] + x[2]^2;

        res = evaluateLogMvDensity(f, supp);

        @test res[1] == 10;
        @test res[2] == 18;

    end
end
