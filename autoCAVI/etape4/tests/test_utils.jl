using Test, Distributions

include("../src/utils.jl");

@testset "utils.jl" begin
    
    @testset "findMode(f, x0)" begin
        
        f(x::DenseVector) = -(x[1]-1)^2 - (x[2]-1)^2;

        mode = findMode(f, [0.0, 0.0]);

        @test (mode[1] - 1.0) < .0001;
        @test (mode[2] - 1.0) < .0001;

    end

    
    @testset "fisherVar(f, x)" begin
        
        m = [0.0, 0.0, 0.0];
        Σ = diagm([1.0, 2.0, 3.0]);

        logf(x::DenseVector) = logpdf(MvNormal(m, Σ), x);

        @test fisherVar(logf, m) ≈ Σ;

    end
end