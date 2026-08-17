#!/bin/sh
# DuckDNS에 현재 공인 IP를 알려 도메인이 항상 이 집을 가리키게 한다.
# 집 IP는 ISP가 수시로 바꾸므로 크론으로 주기적으로 실행한다.
#
# 필요한 값은 ~/stock_dashboard/.env 에서 읽는다:
#   DUCKDNS_SUBDOMAIN=내가정한이름     (예: myhynix  ->  myhynix.duckdns.org)
#   DUCKDNS_TOKEN=duckdns.org에서 발급받은 토큰
#
# 사용법(크론): */10 * * * * /home/yulimseo/stock_dashboard/duckdns_update.sh

ENV_FILE="$(dirname "$0")/.env"
LOG_FILE="$(dirname "$0")/logs/duckdns.log"

[ -f "$ENV_FILE" ] || { echo "$(date '+%F %T') [error] .env 없음: $ENV_FILE" >&2; exit 1; }

# .env에서 필요한 두 값만 읽는다 (다른 비밀값은 건드리지 않는다)
DUCKDNS_SUBDOMAIN=$(grep -E '^DUCKDNS_SUBDOMAIN=' "$ENV_FILE" | cut -d= -f2- | tr -d '"'"'"' \r')
DUCKDNS_TOKEN=$(grep -E '^DUCKDNS_TOKEN=' "$ENV_FILE" | cut -d= -f2- | tr -d '"'"'"' \r')

if [ -z "$DUCKDNS_SUBDOMAIN" ] || [ -z "$DUCKDNS_TOKEN" ]; then
    echo "$(date '+%F %T') [error] .env에 DUCKDNS_SUBDOMAIN / DUCKDNS_TOKEN 이 필요합니다" >&2
    exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

# ip= 를 비워 보내면 DuckDNS가 요청을 보낸 쪽의 공인 IP를 자동으로 잡아준다.
RESULT=$(curl -s -m 20 \
    "https://www.duckdns.org/update?domains=${DUCKDNS_SUBDOMAIN}&token=${DUCKDNS_TOKEN}&ip=")

if [ "$RESULT" = "OK" ]; then
    # 매번 남기면 로그가 계속 커지므로 성공은 하루 한 번만 기록한다.
    TODAY=$(date '+%F')
    if ! grep -q "^${TODAY} .*\[ok\]" "$LOG_FILE" 2>/dev/null; then
        echo "$(date '+%F %T') [ok] ${DUCKDNS_SUBDOMAIN}.duckdns.org 갱신됨" >> "$LOG_FILE"
    fi
else
    echo "$(date '+%F %T') [fail] 응답: ${RESULT:-<빈 응답>}" >> "$LOG_FILE"
    exit 1
fi
