#!/bin/sh
TOKEN=$(curl -s -L "https://public.ecr.aws/token?service=public.ecr.aws&scope=repository:q9t5s3a7/vllm_logits-ci-postmerge-repo:pull" | jq -r .token)
if [[ "$BUILDKITE_BRANCH" == "main" ]]; then
    URL="https://public.ecr.aws/v2/q9t5s3a7/vllm_logits-ci-postmerge-repo/manifests/$BUILDKITE_COMMIT"
else
    URL="https://public.ecr.aws/v2/q9t5s3a7/vllm_logits-ci-test-repo/manifests/$BUILDKITE_COMMIT"
fi

TIMEOUT_SECONDS=10

retries=0
while [ $retries -lt 1000 ]; do
    if [ "$(curl -s --max-time "$TIMEOUT_SECONDS" -L -H "Authorization: Bearer $TOKEN" -o /dev/null -w "%{http_code}" "$URL")" -eq 200 ]; then
        exit 0
    fi

    echo "Waiting for image to be available..."

    retries=$((retries + 1))
    sleep 5
done

exit 1
