-- CreateTable
CREATE TABLE "Reminder" (
    "id"        TEXT         NOT NULL,
    "userId"    TEXT         NOT NULL,
    "exercise"  TEXT         NOT NULL,
    "remindAt"  TIMESTAMP(3) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Reminder_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "Reminder_userId_remindAt_idx" ON "Reminder"("userId", "remindAt");

-- AddForeignKey
ALTER TABLE "Reminder"
    ADD CONSTRAINT "Reminder_userId_fkey"
    FOREIGN KEY ("userId") REFERENCES "User"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;
