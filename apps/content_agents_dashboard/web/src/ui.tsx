// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { useEffect, type ReactNode } from "react";

export function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

interface ThemeProviderProps {
  children: ReactNode;
  theme?: "light" | "dark" | "system";
  global?: boolean;
}

export function ThemeProvider({
  children,
  theme = "system",
  global = false,
}: ThemeProviderProps) {
  useEffect(() => {
    if (!global) return;
    const root = document.documentElement;
    const previousDark = root.classList.contains("nv-dark");
    const previousLight = root.classList.contains("nv-light");
    const resolved =
      theme === "system"
        ? window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light"
        : theme;
    root.classList.remove("nv-dark", "nv-light");
    root.classList.add(resolved === "dark" ? "nv-dark" : "nv-light");
    return () => {
      root.classList.remove("nv-dark", "nv-light");
      if (previousDark) root.classList.add("nv-dark");
      if (previousLight) root.classList.add("nv-light");
    };
  }, [global, theme]);

  return <>{children}</>;
}

interface AppBarProps {
  slotStart?: ReactNode;
  slotEnd?: ReactNode;
}

export function AppBar({ slotStart, slotEnd }: AppBarProps) {
  return (
    <header className="ui-app-bar">
      <div className="ui-app-bar-section">{slotStart}</div>
      <div className="ui-app-bar-section">{slotEnd}</div>
    </header>
  );
}

interface ButtonProps
  extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "color"> {
  kind?: "primary" | "secondary" | "tertiary";
  color?: "brand" | "danger";
  size?: "tiny" | "small" | "large";
}

export function Button({
  kind = "primary",
  color,
  size,
  className,
  type = "button",
  ...props
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cx(
        "ui-button",
        `ui-button-${kind}`,
        color && `ui-button-${color}`,
        size && `ui-button-${size}`,
        className,
      )}
      {...props}
    />
  );
}

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  color?: "brand" | "danger" | "red" | "yellow" | "blue" | "green" | "gray";
  kind?: "solid" | "outline";
}

export function Badge({
  color = "gray",
  kind = "solid",
  className,
  ...props
}: BadgeProps) {
  return (
    <span
      className={cx("ui-badge", `ui-badge-${color}`, `ui-badge-${kind}`, className)}
      {...props}
    />
  );
}

export function Card({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cx("ui-card", className)} {...props} />;
}

interface TextInputProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "size"> {
  size?: "small";
}

export function TextInput({ className, size, ...props }: TextInputProps) {
  return (
    <input
      className={cx("ui-input", size === "small" && "ui-input-small", className)}
      {...props}
    />
  );
}

interface TextAreaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  size?: "small";
}

export function TextArea({ className, size, ...props }: TextAreaProps) {
  return (
    <textarea
      className={cx(
        "ui-input ui-textarea",
        size === "small" && "ui-input-small",
        className,
      )}
      {...props}
    />
  );
}

interface SelectItem {
  value: string;
  children: ReactNode;
}

interface SelectProps
  extends Omit<
    React.SelectHTMLAttributes<HTMLSelectElement>,
    "children" | "onChange" | "value"
  > {
  items: SelectItem[];
  value: string;
  onValueChange?: (value: string) => void;
}

export function Select({
  className,
  items,
  value,
  onValueChange,
  ...props
}: SelectProps) {
  return (
    <select
      className={cx("ui-input", className)}
      value={value}
      onChange={(event) => onValueChange?.(event.target.value)}
      {...props}
    >
      {items.map((item) => (
        <option key={item.value} value={item.value}>
          {item.children}
        </option>
      ))}
    </select>
  );
}

interface SwitchProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "onChange" | "type"> {
  slotLabel?: ReactNode;
  onCheckedChange?: (checked: boolean) => void;
}

export function Switch({
  slotLabel,
  checked,
  onCheckedChange,
  className,
  ...props
}: SwitchProps) {
  return (
    <label className={cx("ui-switch", className)}>
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        onChange={(event) => onCheckedChange?.(event.target.checked)}
        {...props}
      />
      <span className="ui-switch-track" aria-hidden="true">
        <span className="ui-switch-thumb" />
      </span>
      <span className="ui-switch-label">{slotLabel}</span>
    </label>
  );
}

interface TooltipProps {
  children: ReactNode;
  slotContent: string;
}

export function Tooltip({ children, slotContent }: TooltipProps) {
  return (
    <span className="ui-tooltip" title={slotContent}>
      {children}
    </span>
  );
}
