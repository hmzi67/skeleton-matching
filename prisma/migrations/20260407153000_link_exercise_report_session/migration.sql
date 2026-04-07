-- Link ExerciseReport rows to ExerciseSession rows so report views can load
-- full per-set / per-rep session details.
ALTER TABLE "ExerciseReport"
    ADD COLUMN IF NOT EXISTS "sessionId" TEXT;

CREATE INDEX IF NOT EXISTS "ExerciseReport_sessionId_idx"
    ON "ExerciseReport"("sessionId");

ALTER TABLE "ExerciseReport"
    ADD CONSTRAINT "ExerciseReport_sessionId_fkey"
    FOREIGN KEY ("sessionId") REFERENCES "ExerciseSession"("id")
    ON DELETE SET NULL ON UPDATE CASCADE;
