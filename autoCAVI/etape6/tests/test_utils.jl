using Test, Optim, Distributions, LinearAlgebra
using Distributions:loglikelihood

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


    @testset "flatten(mat)" begin
        
        mat = [1 2 3; 4 5 6; 7 8 9];

        vector = flatten(mat);

        @test vector[2] == 2;
        @test vector[6] == 6;
        @test length(vector) == 9;

    end


    @testset "vecToMatrix(vector)" begin
        
        vector = [1, 2, 3, 4, 5, 6, 7, 8, 9];

        mat = vecToMatrix(vector);

        @test mat[1, 2] == 2;
        @test mat[2, 3] == 6;
        @test size(mat) == (3, 3);

    end
end