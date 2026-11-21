"""
Sharing models for public and internal recording shares.

This module defines models for sharing recordings both publicly (via links)
and internally (between users).
"""

import os
import secrets
from datetime import datetime
from src.database import db


# Get sharing configuration from environment
SHOW_USERNAMES_IN_UI = os.environ.get('SHOW_USERNAMES_IN_UI', 'false').lower() == 'true'


class Share(db.Model):
    """Public sharing via shareable links."""

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(32), unique=True, nullable=False, default=lambda: secrets.token_urlsafe(16))
    recording_id = db.Column(db.Integer, db.ForeignKey('recording.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    share_summary = db.Column(db.Boolean, default=True)
    share_notes = db.Column(db.Boolean, default=True)

    user = db.relationship('User', backref=db.backref('shares', lazy=True, cascade='all, delete-orphan'))
    recording = db.relationship('Recording', backref=db.backref('shares', lazy=True, cascade='all, delete-orphan'))

    def to_dict(self):
        """Convert model to dictionary representation."""
        return {
            'id': self.id,
            'public_id': self.public_id,
            'recording_id': self.recording_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'share_summary': self.share_summary,
            'share_notes': self.share_notes,
            'recording_title': self.recording.title if self.recording else "N/A",
            'audio_available': self.recording.audio_deleted_at is None if self.recording else True
        }


class InternalShare(db.Model):
    """Tracks internal sharing of recordings between users (independent of teams)."""

    __tablename__ = 'internal_share'

    id = db.Column(db.Integer, primary_key=True)
    recording_id = db.Column(db.Integer, db.ForeignKey('recording.id', ondelete='CASCADE'), nullable=False)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)  # User who shared
    shared_with_user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)  # User it was shared with

    # Permissions
    can_edit = db.Column(db.Boolean, default=False)  # Can edit notes/metadata
    can_reshare = db.Column(db.Boolean, default=False)  # Can share with others

    # Source tracking for share cleanup
    source_type = db.Column(db.String(20), default='manual')  # 'manual', 'group_tag', or 'group_folder'
    source_tag_id = db.Column(db.Integer, db.ForeignKey('tag.id', ondelete='SET NULL'), nullable=True)
    # The group folder that auto-shared this recording (when source_type='group_folder').
    # Lets a recipient be shown only the share-reason folder, not the owner's other metadata (#314).
    source_folder_id = db.Column(db.Integer, db.ForeignKey('folder.id', ondelete='SET NULL'), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship for source tag
    source_tag = db.relationship('Tag', foreign_keys=[source_tag_id], backref=db.backref('created_shares', lazy=True))

    # Relationships
    recording = db.relationship('Recording', backref=db.backref('internal_shares', lazy=True, cascade='all, delete-orphan'))
    owner = db.relationship('User', foreign_keys=[owner_id], backref=db.backref('shared_recordings', lazy=True, cascade='all, delete'))
    shared_with = db.relationship('User', foreign_keys=[shared_with_user_id], backref=db.backref('received_shares', lazy=True, cascade='all, delete'))

    # Unique constraint: can't share same recording with same user twice
    __table_args__ = (db.UniqueConstraint('recording_id', 'shared_with_user_id', name='unique_recording_share'),)

    def to_dict(self):
        """Convert model to dictionary representation."""
        return {
            'id': self.id,
            'recording_id': self.recording_id,
            'owner_id': self.owner_id,
            'owner_username': self.owner.username if SHOW_USERNAMES_IN_UI else None,
            'user_id': self.shared_with_user_id,  # For frontend compatibility
            'username': self.shared_with.username,  # Always include username
            'can_edit': self.can_edit,
            'can_reshare': self.can_reshare,
            'source_type': self.source_type,
            'source_tag_id': self.source_tag_id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    @staticmethod
    def get_user_max_permissions(recording, user):
        """
        Get the maximum permissions a user can grant for a recording.

        Args:
            recording: Recording object
            user: User object attempting to share

        Returns:
            Dict with 'can_edit' and 'can_reshare' boolean flags
        """
        # Owner has unlimited permissions
        if recording.user_id == user.id:
            return {'can_edit': True, 'can_reshare': True}

        # Get user's share for this recording
        user_share = InternalShare.query.filter_by(
            recording_id=recording.id,
            shared_with_user_id=user.id
        ).first()

        if not user_share:
            # User has no access
            return {'can_edit': False, 'can_reshare': False}

        # User can only grant what they have
        return {
            'can_edit': user_share.can_edit,
            'can_reshare': user_share.can_reshare
        }

    @staticmethod
    def validate_reshare_permissions(recording, grantor_user, requested_permissions):
        """
        Validate that a user can grant the requested permissions.

        Args:
            recording: Recording object being shared
            grantor_user: User attempting to share (current_user)
            requested_permissions: Dict with 'can_edit' and 'can_reshare' flags

        Returns:
            Tuple of (is_valid: bool, error_message: str or None)
        """
        # Owner can grant anything
        if recording.user_id == grantor_user.id:
            return True, None

        # Get grantor's permissions
        max_permissions = InternalShare.get_user_max_permissions(recording, grantor_user)

        # Validate edit permission
        if requested_permissions.get('can_edit', False) and not max_permissions['can_edit']:
            return False, "You cannot grant edit permission because you do not have edit access"

        # Validate reshare permission
        if requested_permissions.get('can_reshare', False) and not max_permissions['can_reshare']:
            return False, "You cannot grant reshare permission because you do not have reshare access"

        return True, None

    @staticmethod
    def is_share_valid_for_user(share, recording=None):
        """Validate that an InternalShare is still backed by active group membership
        and source attachment when derived from a group tag or group folder.

        Returns:
            Tuple of (is_valid: bool, effective_can_edit: bool)
        """
        if not share:
            return False, False

        if share.source_type == 'group_tag':
            if not share.source_tag_id:
                return False, False
            from src.models.organization import GroupMembership, RecordingTag, Tag
            tag = db.session.get(Tag, share.source_tag_id)
            if not tag or not tag.group_id:
                return False, False
            attached = RecordingTag.query.filter_by(
                recording_id=share.recording_id,
                tag_id=share.source_tag_id
            ).first()
            if not attached:
                return False, False
            membership = GroupMembership.query.filter_by(
                group_id=tag.group_id,
                user_id=share.shared_with_user_id
            ).first()
            if not membership:
                return False, False
            if not tag.auto_share_on_apply:
                if not (tag.share_with_group_lead and membership.role == 'admin'):
                    return False, False
            return True, bool(share.can_edit and membership.role == 'admin')

        if share.source_type == 'group_folder':
            if not share.source_folder_id:
                return False, False
            from src.models.organization import Folder, GroupMembership
            from src.models.recording import Recording
            folder = db.session.get(Folder, share.source_folder_id)
            if not folder or not folder.group_id:
                return False, False
            rec = recording if recording is not None else db.session.get(Recording, share.recording_id)
            if not rec or rec.folder_id != share.source_folder_id:
                return False, False
            membership = GroupMembership.query.filter_by(
                group_id=folder.group_id,
                user_id=share.shared_with_user_id
            ).first()
            if not membership:
                return False, False
            if not folder.auto_share_on_apply:
                if not (folder.share_with_group_lead and membership.role == 'admin'):
                    return False, False
            return True, bool(share.can_edit and membership.role == 'admin')

        return True, bool(share.can_edit)

    @staticmethod
    def get_valid_shared_recording_ids(user_id):
        """Return recording IDs shared with user_id via currently valid InternalShare rows."""
        shares = InternalShare.query.filter_by(shared_with_user_id=user_id).all()
        valid_ids = set()
        for share in shares:
            is_valid, _ = InternalShare.is_share_valid_for_user(share)
            if is_valid:
                valid_ids.add(share.recording_id)
        return valid_ids

    @staticmethod
    def revoke_for_group_member(group_id, user_id):
        """Delete all group-derived shares held by user_id from group_id."""
        from src.models.organization import Folder, Tag
        tag_ids = [t.id for t in Tag.query.filter_by(group_id=group_id).all()]
        folder_ids = [f.id for f in Folder.query.filter_by(group_id=group_id).all()]
        deleted = 0
        if tag_ids:
            deleted += InternalShare.query.filter(
                InternalShare.shared_with_user_id == user_id,
                InternalShare.source_type == 'group_tag',
                InternalShare.source_tag_id.in_(tag_ids),
            ).delete(synchronize_session=False)
        if folder_ids:
            deleted += InternalShare.query.filter(
                InternalShare.shared_with_user_id == user_id,
                InternalShare.source_type == 'group_folder',
                InternalShare.source_folder_id.in_(folder_ids),
            ).delete(synchronize_session=False)
        return deleted

    @staticmethod
    def sync_for_group_member_role(group_id, user_id, new_role):
        """Downgrade or revoke group-derived shares when a member's role changes."""
        if new_role == 'admin':
            return 0
        from src.models.organization import Folder, Tag
        updated = 0
        for tag in Tag.query.filter_by(group_id=group_id).all():
            q = InternalShare.query.filter_by(
                shared_with_user_id=user_id,
                source_type='group_tag',
                source_tag_id=tag.id,
            )
            if tag.auto_share_on_apply:
                updated += q.update({'can_edit': False}, synchronize_session=False)
            else:
                updated += q.delete(synchronize_session=False)
        for folder in Folder.query.filter_by(group_id=group_id).all():
            q = InternalShare.query.filter_by(
                shared_with_user_id=user_id,
                source_type='group_folder',
                source_folder_id=folder.id,
            )
            if folder.auto_share_on_apply:
                updated += q.update({'can_edit': False}, synchronize_session=False)
            else:
                updated += q.delete(synchronize_session=False)
        return updated

    @staticmethod
    def revoke_for_group(group_id):
        """Delete all group-derived shares originating from tags or folders in group_id."""
        from src.models.organization import Folder, Tag
        tag_ids = [t.id for t in Tag.query.filter_by(group_id=group_id).all()]
        folder_ids = [f.id for f in Folder.query.filter_by(group_id=group_id).all()]
        deleted = 0
        if tag_ids:
            deleted += InternalShare.query.filter(
                InternalShare.source_type == 'group_tag',
                InternalShare.source_tag_id.in_(tag_ids),
            ).delete(synchronize_session=False)
        if folder_ids:
            deleted += InternalShare.query.filter(
                InternalShare.source_type == 'group_folder',
                InternalShare.source_folder_id.in_(folder_ids),
            ).delete(synchronize_session=False)
        return deleted

    @staticmethod
    def revoke_for_tag(tag_id):
        """Delete all group_tag shares derived from tag_id."""
        return InternalShare.query.filter_by(
            source_type='group_tag',
            source_tag_id=tag_id,
        ).delete(synchronize_session=False)

    @staticmethod
    def revoke_for_folder(folder_id, recording_id=None):
        """Delete group_folder shares derived from folder_id (optionally for one recording)."""
        q = InternalShare.query.filter_by(
            source_type='group_folder',
            source_folder_id=folder_id,
        )
        if recording_id is not None:
            q = q.filter_by(recording_id=recording_id)
        return q.delete(synchronize_session=False)

    @staticmethod
    def find_cascade_revocation_targets(recording, primary_share):
        """Find all transitive downstream InternalShare records to cascade-revoke
        when primary_share is removed, preserving shares whose grantor retains an
        alternate valid access path from the recording owner or unaffected roots.
        """
        from collections import deque

        all_shares = InternalShare.query.filter_by(
            recording_id=primary_share.recording_id
        ).all()
        remaining = [s for s in all_shares if s.id != primary_share.id]

        out_edges = {}
        for s in remaining:
            out_edges.setdefault(s.owner_id, []).append(s)

        # 1. Collect all users in the downstream subgraph rooted at the revoked recipient.
        downstream_users = {primary_share.shared_with_user_id}
        q = deque([primary_share.shared_with_user_id])
        while q:
            uid = q.popleft()
            for edge in out_edges.get(uid, ()):
                nxt = edge.shared_with_user_id
                if nxt not in downstream_users:
                    downstream_users.add(nxt)
                    q.append(nxt)

        # 2. Compute reachability from valid unaffected roots over the remaining graph.
        roots = set()
        if recording is not None and recording.user_id is not None:
            roots.add(recording.user_id)
        for s in remaining:
            if s.owner_id not in downstream_users:
                roots.add(s.owner_id)

        reachable = set(roots)
        rq = deque(roots)
        while rq:
            uid = rq.popleft()
            for edge in out_edges.get(uid, ()):
                nxt = edge.shared_with_user_id
                if nxt not in reachable:
                    reachable.add(nxt)
                    rq.append(nxt)

        # 3. Revoke every downstream share whose grantor no longer has valid access.
        return [
            s for s in remaining
            if s.owner_id in downstream_users and s.owner_id not in reachable
        ]

    @staticmethod
    def find_downstream_shares(recording_id, user_id):
        """
        Find all shares created by a specific user for a recording.
        Used for cascade revocation.

        Args:
            recording_id: ID of the recording
            user_id: ID of the user whose downstream shares to find

        Returns:
            List of InternalShare objects
        """
        return InternalShare.query.filter_by(
            recording_id=recording_id,
            owner_id=user_id
        ).all()

    @staticmethod
    def has_alternate_access_path(recording_id, user_id, excluding_grantor_id=None):
        """
        Check if a user has alternate access to a recording through other shares.
        Used to prevent cascade revocation when user has multiple access paths (diamond pattern).

        Args:
            recording_id: ID of the recording
            user_id: ID of the user to check
            excluding_grantor_id: Exclude shares from this grantor (the one being revoked)

        Returns:
            Boolean - True if user has alternate access path
        """
        query = InternalShare.query.filter(
            InternalShare.recording_id == recording_id,
            InternalShare.shared_with_user_id == user_id
        )

        if excluding_grantor_id is not None:
            query = query.filter(InternalShare.owner_id != excluding_grantor_id)

        return query.count() > 0


class SharedRecordingState(db.Model):
    """Tracks per-user state for shared recordings (notes, highlights, etc)."""

    __tablename__ = 'shared_recording_state'

    id = db.Column(db.Integer, primary_key=True)
    recording_id = db.Column(db.Integer, db.ForeignKey('recording.id', ondelete='CASCADE'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)

    # User-specific state
    personal_notes = db.Column(db.Text, nullable=True)  # Private notes only this user can see
    is_inbox = db.Column(db.Boolean, default=True)  # User's personal inbox status
    is_highlighted = db.Column(db.Boolean, default=False)  # User's personal highlight/favorite status
    last_viewed = db.Column(db.DateTime, default=datetime.utcnow)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    recording = db.relationship('Recording', backref=db.backref('user_states', lazy=True, cascade='all, delete-orphan'))
    user = db.relationship('User', backref=db.backref('recording_states', lazy=True, cascade='all, delete-orphan'))

    # Unique constraint: one state per user per recording
    __table_args__ = (db.UniqueConstraint('recording_id', 'user_id', name='unique_user_recording_state'),)

    def to_dict(self):
        """Convert model to dictionary representation."""
        return {
            'id': self.id,
            'recording_id': self.recording_id,
            'user_id': self.user_id,
            'personal_notes': self.personal_notes,
            'is_inbox': self.is_inbox,
            'is_highlighted': self.is_highlighted,
            'last_viewed': self.last_viewed.isoformat() if self.last_viewed else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
