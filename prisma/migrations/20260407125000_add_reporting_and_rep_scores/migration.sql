-- Backfill missing columns on ExerciseReport that were declared in
-- schema.prisma but never migrated.
ALTER TABLE "ExerciseReport"
    ADD COLUMN IF NOT EXISTS "repScores"      TEXT NOT NULL DEFAULT '[]';
ALTER TABLE "ExerciseReport"
    ADD COLUMN IF NOT EXISTS "repJointIssues" TEXT NOT NULL DEFAULT '[]';

-- CreateTable
CREATE TABLE "ExerciseSession" (
    "id"           TEXT         NOT NULL,
    "userId"       TEXT         NOT NULL,
    "exerciseName" TEXT         NOT NULL,
    "videoId"      TEXT,
    "startedAt"    TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completedAt"  TIMESTAMP(3),
    "completed"    BOOLEAN      NOT NULL DEFAULT false,

    CONSTRAINT "ExerciseSession_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SetReport" (
    "id"              TEXT             NOT NULL,
    "sessionId"       TEXT             NOT NULL,
    "setNumber"       INTEGER          NOT NULL,
    "overallAccuracy" DOUBLE PRECISION NOT NULL,
    "grade"           TEXT             NOT NULL,
    "reportJson"      JSONB            NOT NULL,
    "createdAt"       TIMESTAMP(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "SetReport_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SessionReport" (
    "id"              TEXT             NOT NULL,
    "sessionId"       TEXT             NOT NULL,
    "overallAccuracy" DOUBLE PRECISION NOT NULL,
    "grade"           TEXT             NOT NULL,
    "completed"       BOOLEAN          NOT NULL,
    "reportJson"      JSONB            NOT NULL,
    "createdAt"       TIMESTAMP(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "SessionReport_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "SessionReport_sessionId_key" ON "SessionReport"("sessionId");

-- AddForeignKey
ALTER TABLE "ExerciseSession"
    ADD CONSTRAINT "ExerciseSession_userId_fkey"
    FOREIGN KEY ("userId") REFERENCES "User"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SetReport"
    ADD CONSTRAINT "SetReport_sessionId_fkey"
    FOREIGN KEY ("sessionId") REFERENCES "ExerciseSession"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SessionReport"
    ADD CONSTRAINT "SessionReport_sessionId_fkey"
    FOREIGN KEY ("sessionId") REFERENCES "ExerciseSession"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;
