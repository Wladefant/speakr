"""
Transcription usage tracking service for monitoring audio transcription consumption and budget enforcement.
"""

import logging
import threading
from datetime import date, datetime, timedelta
from typing import Tuple, Optional, Dict, List

from sqlalchemy import func, extract

from src.database import db
from src.models.transcription_usage import TranscriptionUsage
from src.models.user import User

logger = logging.getLogger(__name__)


# Pricing configuration per connector/model (dollars per minute)
TRANSCRIPTION_PRICING = {
    'openai_whisper': {
        'whisper-1': 0.006,  # $0.006/min
        'default': 0.006,
    },
    'openai_transcribe': {
        'gpt-4o-transcribe': 0.006,      # $0.006/min
        'gpt-4o-mini-transcribe': 0.003,  # $0.003/min
        'gpt-4o-mini-transcribe-2025-12-15': 0.003,
        'gpt-4o-transcribe-diarize': 0.006,
        'default': 0.006,
    },
    'asr_endpoint': {
        'default': 0.0,  # Self-hosted = free
    },
}


def get_transcription_cost_per_minute(connector_type: str, model_name: str = None) -> float:
    """
    Get the cost per minute for a given connector and model.

    Args:
        connector_type: The connector provider name
        model_name: The specific model (optional)

    Returns:
        Cost per minute in dollars
    """
    connector_pricing = TRANSCRIPTION_PRICING.get(connector_type, {})

    if model_name and model_name in connector_pricing:
        return connector_pricing[model_name]

    # Fall back to 'default' pricing for the connector
    return connector_pricing.get('default', 0.0)


class TranscriptionTracker:
    def __init__(self):
        self._lock = threading.RLock()
        self._inflight_seconds: Dict[int, float] = {}

    """Service for recording and checking transcription usage."""

    CONNECTOR_TYPES = [
        'openai_whisper',
        'openai_transcribe',
        'asr_endpoint',
    ]

    def record_usage(
        self,
        user_id: int,
        connector_type: str,
        audio_duration_seconds: int,
        model_name: str = None,
        estimated_cost: float = None
    ):
        """
        Record transcription usage - upserts into daily aggregate.

        Args:
            user_id: User ID who made the request
            connector_type: Type of connector (openai_whisper, openai_transcribe, asr_endpoint)
            audio_duration_seconds: Duration of audio transcribed in seconds
            model_name: Name of the model used
            estimated_cost: Pre-calculated cost (if None, calculated from pricing config)
        """
        try:
            today = date.today()

            # Calculate cost if not provided
            if estimated_cost is None:
                cost_per_minute = get_transcription_cost_per_minute(connector_type, model_name)
                estimated_cost = (audio_duration_seconds / 60.0) * cost_per_minute

            # Find or create today's record for this user/connector
            usage = TranscriptionUsage.query.filter_by(
                user_id=user_id,
                date=today,
                connector_type=connector_type
            ).first()

            if usage:
                # Update existing record
                usage.audio_duration_seconds += audio_duration_seconds
                usage.estimated_cost += estimated_cost
                usage.request_count += 1
                if model_name:
                    usage.model_name = model_name  # Update to latest model used
            else:
                # Create new record
                usage = TranscriptionUsage(
                    user_id=user_id,
                    date=today,
                    connector_type=connector_type,
                    audio_duration_seconds=audio_duration_seconds,
                    request_count=1,
                    model_name=model_name,
                    estimated_cost=estimated_cost or 0.0
                )
                db.session.add(usage)

            db.session.commit()
            logger.debug(f"Recorded {audio_duration_seconds}s transcription for user {user_id}, connector {connector_type}")
            return usage

        except Exception as e:
            logger.error(f"Failed to record transcription usage: {e}")
            db.session.rollback()
            return None

    def get_monthly_usage(self, user_id: int, year: int = None, month: int = None) -> int:
        """Get total seconds transcribed by a user in a given month."""
        if year is None:
            year = date.today().year
        if month is None:
            month = date.today().month

        result = db.session.query(func.sum(TranscriptionUsage.audio_duration_seconds)).filter(
            TranscriptionUsage.user_id == user_id,
            extract('year', TranscriptionUsage.date) == year,
            extract('month', TranscriptionUsage.date) == month
        ).scalar()

        return result or 0

    def get_monthly_cost(self, user_id: int, year: int = None, month: int = None) -> float:
        """Get total estimated cost for a user in a given month."""
        if year is None:
            year = date.today().year
        if month is None:
            month = date.today().month

        result = db.session.query(func.sum(TranscriptionUsage.estimated_cost)).filter(
            TranscriptionUsage.user_id == user_id,
            extract('year', TranscriptionUsage.date) == year,
            extract('month', TranscriptionUsage.date) == month
        ).scalar()

        return result or 0.0

    def _check_budget_unlocked(self, user_id: int, additional_seconds: float = 0.0) -> Tuple[bool, float, Optional[str]]:
        user = db.session.get(User, user_id)
        if not user or not user.monthly_transcription_budget:
            return (True, 0, None)  # No budget = unlimited

        current_usage = self.get_monthly_usage(user_id)
        inflight = self._inflight_seconds.get(user_id, 0.0)
        requested = max(0.0, float(additional_seconds or 0.0))
        effective_usage = current_usage + inflight + requested
        budget = user.monthly_transcription_budget
        percentage = (effective_usage / budget) * 100

        if current_usage + inflight >= budget or (requested > 0 and effective_usage > budget):
            minutes_used = int((current_usage + inflight) // 60)
            minutes_budget = budget // 60
            return (False, percentage,
                    f"Monthly transcription budget exceeded ({minutes_used}/{minutes_budget} minutes). Contact admin for more time.")
        elif percentage >= 80:
            return (True, percentage,
                    f"Warning: {percentage:.1f}% of monthly transcription budget used.")
        else:
            return (True, percentage, None)

    def check_budget(self, user_id: int, additional_seconds: float = 0.0) -> Tuple[bool, float, Optional[str]]:
        """Check if user is within transcription budget (including in-flight reservations and optional requested duration)."""
        try:
            with self._lock:
                return self._check_budget_unlocked(user_id, additional_seconds=additional_seconds)
        except Exception as e:
            logger.error(f"Failed to check transcription budget for user {user_id}: {e}")
            return (True, 0, None)

    def reserve_budget(self, user_id: Optional[int], additional_seconds: float = 0.0) -> Tuple[bool, float, Optional[str]]:
        """Atomically verify budget and reserve in-flight seconds for a user."""
        if user_id is None:
            return (True, 0, None)
        requested = max(0.0, float(additional_seconds or 0.0))
        try:
            with self._lock:
                allowed, pct, msg = (self.check_budget(user_id, additional_seconds=requested) if requested > 0 else self.check_budget(user_id))
                if allowed and requested > 0:
                    self._inflight_seconds[user_id] = self._inflight_seconds.get(user_id, 0.0) + requested
                return (allowed, pct, msg)
        except Exception as e:
            logger.error(f"Failed to reserve transcription budget for user {user_id}: {e}")
            return (True, 0, None)

    def release_reservation(self, user_id: Optional[int], reserved_seconds: float = 0.0) -> None:
        """Release previously reserved in-flight transcription seconds."""
        if user_id is None or not reserved_seconds or reserved_seconds <= 0:
            return
        with self._lock:
            remaining = self._inflight_seconds.get(user_id, 0.0) - float(reserved_seconds)
            if remaining <= 0:
                self._inflight_seconds.pop(user_id, None)
            else:
                self._inflight_seconds[user_id] = remaining

    def get_daily_stats(self, days: int = 30, user_id: int = None) -> List[Dict]:
        """Get daily transcription usage for charts."""
        start_date = date.today() - timedelta(days=days - 1)

        query = db.session.query(
            TranscriptionUsage.date,
            TranscriptionUsage.connector_type,
            func.sum(TranscriptionUsage.audio_duration_seconds).label('seconds'),
            func.sum(TranscriptionUsage.estimated_cost).label('cost')
        ).filter(TranscriptionUsage.date >= start_date)

        if user_id:
            query = query.filter(TranscriptionUsage.user_id == user_id)

        results = query.group_by(TranscriptionUsage.date, TranscriptionUsage.connector_type).all()

        # Organize by date
        stats = {}
        for r in results:
            date_str = r.date.isoformat()
            if date_str not in stats:
                stats[date_str] = {'date': date_str, 'total_seconds': 0, 'total_minutes': 0, 'cost': 0.0, 'by_connector': {}}
            stats[date_str]['total_seconds'] += r.seconds or 0
            stats[date_str]['total_minutes'] = stats[date_str]['total_seconds'] // 60
            stats[date_str]['cost'] += r.cost or 0
            stats[date_str]['by_connector'][r.connector_type] = {
                'seconds': r.seconds or 0,
                'minutes': (r.seconds or 0) // 60
            }

        # Fill in missing dates with zeros
        all_dates = []
        current = start_date
        while current <= date.today():
            date_str = current.isoformat()
            if date_str not in stats:
                stats[date_str] = {'date': date_str, 'total_seconds': 0, 'total_minutes': 0, 'cost': 0.0, 'by_connector': {}}
            all_dates.append(date_str)
            current += timedelta(days=1)

        return [stats[d] for d in sorted(all_dates)]

    def get_monthly_stats(self, months: int = 12) -> List[Dict]:
        """Get monthly transcription usage for charts."""
        results = db.session.query(
            extract('year', TranscriptionUsage.date).label('year'),
            extract('month', TranscriptionUsage.date).label('month'),
            func.sum(TranscriptionUsage.audio_duration_seconds).label('seconds'),
            func.sum(TranscriptionUsage.estimated_cost).label('cost')
        ).group_by('year', 'month').order_by('year', 'month').all()

        # Get last N months
        monthly_data = [
            {
                'year': int(r.year),
                'month': int(r.month),
                'seconds': r.seconds or 0,
                'minutes': (r.seconds or 0) // 60,
                'cost': r.cost or 0
            }
            for r in results
        ]

        return monthly_data[-months:] if len(monthly_data) > months else monthly_data

    def get_user_stats(self) -> List[Dict]:
        """Get per-user transcription usage breakdown for current month."""
        today = date.today()

        results = db.session.query(
            User.id,
            User.username,
            User.monthly_transcription_budget,
            func.sum(TranscriptionUsage.audio_duration_seconds).label('usage'),
            func.sum(TranscriptionUsage.estimated_cost).label('cost')
        ).outerjoin(
            TranscriptionUsage,
            (User.id == TranscriptionUsage.user_id) &
            (extract('year', TranscriptionUsage.date) == today.year) &
            (extract('month', TranscriptionUsage.date) == today.month)
        ).group_by(User.id).all()

        return [
            {
                'user_id': r.id,
                'username': r.username,
                'monthly_budget_seconds': r.monthly_transcription_budget,
                'monthly_budget_minutes': (r.monthly_transcription_budget // 60) if r.monthly_transcription_budget else None,
                'current_usage_seconds': r.usage or 0,
                'current_usage_minutes': (r.usage or 0) // 60,
                'cost': r.cost or 0,
                'percentage': ((r.usage or 0) / r.monthly_transcription_budget * 100)
                              if r.monthly_transcription_budget else 0
            }
            for r in results
        ]

    def get_today_usage(self, user_id: int = None) -> Dict:
        """Get today's transcription usage."""
        today = date.today()

        query = db.session.query(
            func.sum(TranscriptionUsage.audio_duration_seconds).label('seconds'),
            func.sum(TranscriptionUsage.estimated_cost).label('cost')
        ).filter(TranscriptionUsage.date == today)

        if user_id:
            query = query.filter(TranscriptionUsage.user_id == user_id)

        result = query.first()

        return {
            'seconds': result.seconds or 0,
            'minutes': (result.seconds or 0) // 60,
            'cost': result.cost or 0
        }


# Singleton instance
transcription_tracker = TranscriptionTracker()
