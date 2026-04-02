-- CreateTable
CREATE TABLE "ExerciseReport" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "exercise" TEXT NOT NULL,
    "referenceVideoId" TEXT NOT NULL,
    "durationSeconds" INTEGER NOT NULL,
    "totalFrames" INTEGER NOT NULL,
    "repCount" INTEGER NOT NULL,
    "avgScore" DOUBLE PRECISION NOT NULL,
    "minScore" DOUBLE PRECISION NOT NULL,
    "maxScore" DOUBLE PRECISION NOT NULL,
    "scoreHistory" TEXT NOT NULL,
    "phaseHistory" TEXT NOT NULL,
    "jointIssues" TEXT NOT NULL,
    "boneIssues" TEXT NOT NULL,
    "feedbackSummary" TEXT NOT NULL,
    "overallGrade" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ExerciseReport_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "ExerciseReport" ADD CONSTRAINT "ExerciseReport_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
