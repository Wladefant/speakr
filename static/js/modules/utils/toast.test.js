import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { showToast } from './toast.js';

function createFakeElement(tag) {
    const el = {
        tagName: tag.toUpperCase(),
        className: '',
        style: {},
        children: [],
        parentNode: null,
        textContent: '',
        classList: {
            add: vi.fn(),
            remove: vi.fn(),
        },
        addEventListener: vi.fn(),
        appendChild(child) {
            child.parentNode = el;
            el.children.push(child);
            return child;
        },
        removeChild(child) {
            const idx = el.children.indexOf(child);
            if (idx >= 0) el.children.splice(idx, 1);
            child.parentNode = null;
        },
    };
    Object.defineProperty(el, 'innerHTML', {
        get() {
            return '';
        },
        set() {
            throw new Error('innerHTML assignment is forbidden in showToast');
        },
    });
    return el;
}

describe('showToast XSS safety', () => {
    let container;

    beforeEach(() => {
        vi.useFakeTimers();
        container = createFakeElement('div');
        globalThis.document = {
            getElementById: (id) => (id === 'toastContainer' ? container : null),
            createElement: (tag) => createFakeElement(tag),
        };
        globalThis.requestAnimationFrame = (cb) => cb();
    });

    afterEach(() => {
        vi.useRealTimers();
        delete globalThis.document;
        delete globalThis.requestAnimationFrame;
    });

    it('renders HTML payloads via textContent without assigning innerHTML', () => {
        const payload = '<img src=x onerror=alert(1)>';
        showToast(`Moved to ${payload}`, 'fa-folder', 3000, 'success');

        expect(container.children).toHaveLength(1);
        const toast = container.children[0];
        expect(toast.children).toHaveLength(2);
        const [iconEl, msgEl] = toast.children;
        expect(iconEl.tagName).toBe('I');
        expect(iconEl.className).toBe('fas fa-folder');
        expect(msgEl.tagName).toBe('SPAN');
        expect(msgEl.textContent).toBe(`Moved to ${payload}`);
        expect(msgEl.children).toHaveLength(0);
    });
});
